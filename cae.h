/*
 * CAE (Complexity-Aware Encoding) Filter for FFmpeg
 *
 * This filter analyzes video frames to detect scene changes based on
 * essential metrics (DCT Energy, Sobel Energy, SAD). Upon detecting a
 * scene change, it computes additional metrics (Color Variance, Blockiness,
 * Entropy) to refine the prediction of the Constant Rate Factor (CRF)
 * and dynamically adjusts VBV settings for optimal encoding.
 *
 * Author: Zaki Ahmed
 * Date:2024-10-24
 */

#ifndef AVFILTER_CAE_H
#define AVFILTER_CAE_H

#include <libavutil/opt.h>              // For options parsing and AVClass
#include <libavutil/pixfmt.h>           // For pixel format definitions (e.g., AV_PIX_FMT_YUV420P)
#include <libavutil/frame.h>            // For AVFrame structure
#include <libavfilter/avfilter.h>        // For AVFilterContext and general filter definitions
#include <libavfilter/buffersink.h>     // For buffersink filter handling
#include <libavfilter/buffersrc.h>      // For buffersrc filter handling
#include <libavutil/log.h>              // For logging
#include <libavutil/motion_vector.h>    // For motion vector data structures
#include <libavutil/imgutils.h>         // For image utility functions (e.g., for calculating frame size)
#include <pthread.h> // For POSIX threads and mutex


// Include metric calculation headers
#include "calculate_frame_dct_energy.h"   
#include "calculate_frame_sobel.h"           
#include "calculate_frame_sad.h" 

// Normalize each metric and clamp between 0.0 and 1.0
#define CLAMP_TO_UNIT(value) ((value) < 0.0 ? 0.0 : ((value) > 1.0 ? 1.0 : (value)))



#define METADATA_KEY_PREDICTED_CRF "predicted_crf"
#define METADATA_KEY_VBV_MAXRATE "vbv_maxrate"
#define METADATA_KEY_VBV_BUFSIZE "vbv_bufsize"
#define METADATA_KEY_VBV_INITIAL_BUFFER "vbv_initial_buffer"
#define METADATA_KEY_VBV_PEAKRATE "vbv_peakrate"
#define METADATA_KEY_VBV_MINRATE "vbv_minrate"

// Constants and Macros
#define DEFAULT_WINDOW_SIZE 10
#define MAX_WINDOW_SIZE 50
#define DEFAULT_ALPHA 0.5
#define DEFAULT_BETA 5.0
#define DEFAULT_LAMBDA 0.7
#define DEFAULT_GRADUAL_FRAME_COUNT 3
#define DEFAULT_GRADUAL_THRESHOLD_MULTIPLIER 1.0
#define DEFAULT_DCT_BLOCK_SIZE 8
#define DEFAULT_SOBEL_BLOCK_SIZE 3
#define DEFAULT_SAD_BLOCK_SIZE 8
#define MAX_PREV_FRAMES 150  // Adjusted based on GOP size and frame rate
// Constants
#define EDGE_DETECTION_THRESHOLD 128
#define GLCM_MATRIX_SIZE 256

#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

// Frame Metrics Structure
typedef struct {
    double dct_energy;               ///< DCT energy
    float sobel_energy;              ///< Sobel energy
    float sad_energy;                ///< SAD energy
    double motion_vector_magnitude;  ///< Motion vector magnitude
    double edge_density;             ///< Edge density
    double histogram_complexity;     ///< Histogram complexity
    double temporal_difference;      ///< Temporal difference
    double texture_complexity;       ///< Texture complexity

    // Additional metrics
    double color_variance;           ///< Color variance
    double blockiness;               ///< Blockiness
    double entropy;                  ///< Entropy
} FrameMetrics;

// CAE Context Structure
typedef struct CaeContext {
    const AVClass *class;             ///< FFmpeg class for filter options
    AVFrame *prev_frames[MAX_PREV_FRAMES];   ///< Buffer for previous frames
    int num_prev_frames;              ///< Number of frames stored in prev_frames
    int num_frames;                   ///< Number of frames processed

    // Weights for each metric in complexity calculation
    double weights_dct;               ///< Weight for DCT energy
    double weights_sobel;             ///< Weight for Sobel energy
    double weights_sad;               ///< Weight for SAD energy

    // Complexity scores
    double previous_complexity;       ///< Previous frame's complexity
    double smoothed_complexity;       ///< Smoothed complexity using EMA

    // Moving average window for complexity
    double *complexity_window;        ///< Dynamically allocated array for complexity window
    int window_size;                  ///< Size of the moving window
    int window_index;                 ///< Current index in the window
    bool is_window_filled;            ///< Flag indicating if the window is filled

    // Frame Metrics Window
    FrameMetrics *metrics_window;     ///< Sliding window of frame metrics
    int metrics_window_size;          ///< Size of the metrics window
    int metrics_window_index;         ///< Current index in the metrics window
    bool metrics_window_filled;       ///< Flag indicating if the metrics window is filled

    // Scene change detection
    int gradual_change_count;                     ///< Counter for gradual scene changes
    double cumulative_complexity_change;          ///< Cumulative change for gradual scene change detection
    int gradual_frame_count;                      ///< Number of consecutive frames for gradual scene change detection
    double gradual_threshold_multiplier;          ///< Multiplier for cumulative complexity change to detect gradual scene changes

    // Block sizes for metric calculations
    int dct_block_size;            ///< Block size for DCT energy calculation
    int sobel_block_size;          ///< Block size for Sobel energy calculation
    int sad_block_size;            ///< Block size for SAD calculation

    // Configurable parameters
    double alpha;                  ///< Alpha multiplier for threshold calculation
    double beta;                   ///< Beta addition for threshold calculation
    double lambda;                 ///< Lambda for EMA smoothing

    // Dynamic weighting parameters
    double dynamic_weight_dct;     ///< Dynamic weight for DCT energy
    double dynamic_weight_sobel;   ///< Dynamic weight for Sobel energy
    double dynamic_weight_sad;     ///< Dynamic weight for SAD energy

    // Processing time measurement
    double processing_time_sec;    ///< Processing time for the current frame in seconds

    // CRF Values
    int crf;                        ///< Predicted CRF value (integer)
    int current_crf;                ///< Current CRF value, updated per frame (integer)

    // VBV Parameters
    int vbv_maxrate;                ///< Maximum VBV bitrate
    int vbv_bufsize;                ///< VBV buffer size
    int vbv_initial_buffer;         ///< Initial VBV buffer fullness
    int vbv_peakrate;               ///< VBV peak bitrate
    int vbv_minrate;                ///< VBV minimum bitrate
    int vbv_adjust_cooldown;        ///< Frames remaining in cooldown period
    int vbv_cooldown_duration;      ///< Duration of cooldown period (e.g., 30 frames)

    FILE *metrics_file;    ///< File pointer for writing metrics to CSV
    char *metrics_filename; ///< Filename for the CSV output
    pthread_mutex_t mutex; ///< Mutex for thread-safe file writing
} CaeContext;

/**
 * @brief Structure to hold CRF side data.
 * @deprecated Use metadata instead in FFmpeg 2.6
 */
typedef struct {
    double crf; ///< Predicted CRF value
} CRFSideData;

// Function Prototypes
int calculate_and_log_dct_energy(AVFilterContext *ctx, const AVFrame *frame, int dct_size, double *dct_energy);
int calculate_and_log_sobel_energy(AVFilterContext *ctx, const AVFrame *frame, int block_size, float *sobel_energy);
int calculate_and_log_sad(AVFilterContext *ctx, const AVFrame *frame, const AVFrame *prev_frame, int block_size, float *sad);
int calculate_and_log_motion_vector_magnitude(AVFilterContext *ctx, const AVFrame *frame, double *motion_magnitude);
int calculate_and_log_edge_density(AVFilterContext *ctx, const AVFrame *frame, double *edge_density);
int calculate_and_log_histogram_complexity(AVFilterContext *ctx, const AVFrame *frame, double *histogram_complexity);
int calculate_and_log_temporal_difference(AVFilterContext *ctx, const AVFrame *current_frame, const AVFrame *previous_frame, double *temporal_difference);
int calculate_and_log_texture_complexity(AVFilterContext *ctx, const AVFrame *frame, double *texture_complexity);
int calculate_and_log_color_variance(AVFilterContext *ctx, const AVFrame *frame, double *color_variance);
int calculate_and_log_blockiness(AVFilterContext *ctx, const AVFrame *frame, double *blockiness);
int calculate_and_log_entropy(AVFilterContext *ctx, const AVFrame *frame, double *entropy);
int validate_metric(AVFilterContext *ctx, const char *metric_name, double value);
void normalize_metrics(FrameMetrics *metrics);

#endif /* AVFILTER_CAE_H */