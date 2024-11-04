/*
 * CAE (Complexity-Aware Encoding) Filter for FFmpeg
 *
 * Enhanced to include additional metrics (Entropy and Color Variance), advanced normalization,
 * performance optimizations using ARM NEON, enhanced logging, dynamic parameter tuning,
 * and scene change handling.
 *
 * Author: Zaki Ahmed
 * Date: 2024-10-31
 */

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <stddef.h> // For offsetof
#include "libavutil/internal.h"
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include "avfilter.h"
#include "filters.h"
#include "video.h"

#include <libavutil/log.h>
#include <libavutil/frame.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libswscale/swscale.h>
#include <libavutil/mem.h>
#include <pthread.h> // For POSIX threads and mutex
#include <omp.h>      // For OpenMP
#include <time.h>
#include <fftw3.h>    // For DCT computations

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define START_TIMER(name) \
    struct timespec start_##name, end_##name; \
    clock_gettime(CLOCK_MONOTONIC, &start_##name);

#define END_TIMER(name, description) \
    clock_gettime(CLOCK_MONOTONIC, &end_##name); \
    double elapsed_##name = (end_##name.tv_sec - start_##name.tv_sec) * 1e3 + \
                            (end_##name.tv_nsec - start_##name.tv_nsec) / 1e6; \
    av_log(ctx, AV_LOG_DEBUG, "%s: %.3f ms\n", description, elapsed_##name);

#define BLOCK_SIZE 16 // BLOCK SIZE

typedef struct CaeContext {
    const AVClass *class;

    int frame_counter;            // Counts the number of frames processed
    int frame_interval;           // Interval at which frames are processed

    // User-configurable options
    double alpha_complexity;              // Threshold multiplier for complexity
    double alpha_ssim;                   // Threshold multiplier for SSIM
    double alpha_hist;                   // Threshold multiplier for Histogram Difference
    double alpha_dct;                    // Threshold multiplier for DCT Energy
    double alpha_sobel;                  // Threshold multiplier for Sobel Energy
    int window_size;                     // Sliding window size
    int threshold_mode;                  // Mode for thresholding (e.g., 0: Median + alpha*MAD)
    int cooldown_frames;                 // Number of frames to ignore after a scene change
    int required_consecutive_changes;    // Number of consecutive detections to confirm scene change
    double k_threshold;                  // Multiplier for MAD in adaptive threshold calculation
    double max_weight;                   // Maximum allowable weight to prevent domination

    // Internal variables
    double *complexity_window;           // Sliding window for complexity scores (delta_complexity)
    double *ssim_window;                 // Sliding window for SSIM scores (delta_ssim)
    double *hist_window;                 // Sliding window for histogram differences (delta_hist)
    double *dct_window;                  // Sliding window for DCT Energy (delta_dct)
    double *sobel_window;                // Sliding window for Sobel Energy (delta_sobel)
    double *entropy_window;              // Sliding window for Entropy
    double *color_var_window;            // Sliding window for Color Variance
    int window_index;                    // Current index in the window
    bool window_filled;                  // Indicates if the window is fully populated
    double previous_complexity;          // Complexity of the previous frame

    // Scene Change Detection Enhancements
    int current_cooldown;                // Current cooldown counter
    int consecutive_detected;            // Counter for consecutive detections

    // Dynamic Weights for Metrics
    double weight_complexity;
    double weight_ssim;
    double weight_hist;
    double weight_dct;
    double weight_sobel;
    double weight_entropy;
    double weight_color_var;

    // For Dynamic Adaptive Threshold
    double *weighted_sum_window;         // Sliding window for weighted_sum
    double median_weighted_sum;          // Median of weighted_sum_window
    double mad_weighted_sum;             // MAD of weighted_sum_window
    int weighted_sum_window_size;        // Size of weighted_sum_window
    int weighted_sum_window_index;       // Current index in weighted_sum_window
    bool weighted_sum_window_filled;     // Indicates if weighted_sum_window is fully populated

    // FFTW Plan
    fftw_plan dct_plan;                   // FFTW plan for DCT computations
    double *dct_input;                    // Input buffer for DCT
    fftw_complex *dct_output;             // Output buffer for DCT

    // For color space conversion
    struct SwsContext *sws_ctx;
    enum AVPixelFormat src_pix_fmt;
    enum AVPixelFormat dst_pix_fmt;
    int width;
    int height;

    // Previous grayscale frame
    AVFrame *prev_gray_frame;            // AVFrame to store previous grayscale data

    // Additional structures for Histogram
    int hist_prev[256];
    int hist_curr[256];

    // **Adaptive Frame Interval Variables**
    int min_frame_interval;       // Minimum frame interval
    int max_frame_interval;       // Maximum frame interval
    double activity_score;        // Combined activity score based on metrics

    // **New Variables for Normalization**
    double min_complexity, max_complexity;
    double min_ssim, max_ssim;
    double min_hist, max_hist;
    double min_dct, max_dct;
    double min_sobel, max_sobel;
    double min_entropy, max_entropy;
    double min_color_var, max_color_var;

    double crf_exponent; // Exponent for CRF scaling

    double sigmoid_slope;
    double sigmoid_midpoint;

    AVFilterContext *ctx; // Reference to the filter context for logging
} CaeContext;

// Function prototypes
static int init_cae_context(AVFilterContext *ctx);
static void uninit_cae_context(AVFilterContext *ctx);
static av_cold int init_filter(AVFilterContext *ctx);
static av_cold void uninit_filter(AVFilterContext *ctx);
static int filter_frame(AVFilterLink *inlink, AVFrame *frame);
static double compute_SAD(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride);
static double compute_SAD_NEON(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride);
static double compute_SSIM(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride);
static double compute_SSIM_NEON(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride);
static double compute_hist_diff(int *hist1, int *hist2);
static void compute_histogram(const uint8_t *data, int width, int height, int stride, int *hist);
static void compute_histogram_NEON(const uint8_t *data, int width, int height, int stride, int *hist);
static double compute_DCT_energy(CaeContext *s, const uint8_t *data, int width, int height, int stride);
static double compute_Sobel_energy(const uint8_t *data, int width, int height, int stride);
static double compute_Sobel_energy_NEON(const uint8_t *src, int width, int height, int stride);
static double compute_entropy(const uint8_t *data, int width, int height, int stride);
static double compute_entropy_NEON(const uint8_t *data, int width, int height, int stride);
static double compute_color_variance(const uint8_t *data, int width, int height, int stride);
static double compute_color_variance_NEON(const uint8_t *data, int width, int height, int stride);
static bool calculate_mad(const double *data, int size, double median, double *mad);
static bool calculate_median(const double *data, int size, double *median);
static int compare_doubles(const void *a, const void *b);
static int cae_config_props(AVFilterLink *inlink);
static void adjust_weights(CaeContext *s, double delta_complexity, double delta_ssim, double delta_hist, double delta_dct, double delta_sobel, double delta_entropy, double delta_color_var);
static bool calculate_adaptive_threshold(CaeContext *s);
static bool is_finite_double(double value);
static int calculate_dynamic_crf(CaeContext *s, double activity_score);
static int attach_crf_metadata(AVFrame *frame, int crf);


/**
 * @brief Validate if a double value is finite (not NaN or Inf).
 *
 * @param value The value to validate.
 * @return true if finite, false otherwise.
 */
static bool is_finite_double(double value) {
    return !isnan(value) && !isinf(value);
}

/**
 * @brief Comparator for qsort (ascending order)
 */
static int compare_doubles(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;

    // Handle NaN cases
    if (isnan(da) && isnan(db)) return 0;
    if (isnan(da)) return 1; // NaNs are considered greater
    if (isnan(db)) return -1;

    // Handle Infinities
    if (da == db) return 0;
    if (da == INFINITY) return 1;
    if (db == INFINITY) return -1;
    if (da == -INFINITY) return -1;
    if (db == -INFINITY) return 1;

    // Regular comparison
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

/**
 * @brief Function to calculate median with explicit error handling
 */
static bool calculate_median(const double *data, int size, double *median) {
    if (size <= 0 || median == NULL)
        return false;

    double *sorted = malloc(size * sizeof(double));
    if (!sorted)
        return false; // Memory allocation failed

    memcpy(sorted, data, size * sizeof(double));
    qsort(sorted, size, sizeof(double), compare_doubles);

    if (size % 2 == 0)
        *median = (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0;
    else
        *median = sorted[size / 2];

    free(sorted);
    return true;
}

/**
 * @brief Function to calculate Median Absolute Deviation (MAD) with error handling
 */
static bool calculate_mad(const double *data, int size, double median, double *mad) {
    if (size <= 0 || mad == NULL)
        return false;

    double *deviations = malloc(size * sizeof(double));
    if (!deviations)
        return false; // Memory allocation failed

    for (int i = 0; i < size; i++) {
        deviations[i] = fabs(data[i] - median);
    }

    bool success = calculate_median(deviations, size, mad);
    free(deviations);
    return success;
}

/**
 * @brief Function to compute histogram
 */
static void compute_histogram(const uint8_t *data, int width, int height, int stride, int *hist) {
    memset(hist, 0, 256 * sizeof(int));
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const uint8_t *row = data + y * stride;
        for (int x = 0; x < width; x++) {
            #pragma omp atomic
            hist[row[x]]++;
        }
    }
}

/**
 * @brief Function to compute histogram using NEON and private histograms
 */
static void compute_histogram_NEON(const uint8_t *data, int width, int height, int stride, int *hist) {
#ifdef __ARM_NEON
    memset(hist, 0, 256 * sizeof(int));

    int num_threads = omp_get_max_threads();
    int **private_hists = malloc(num_threads * sizeof(int*));
    if (!private_hists) {
        av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory for private histograms.\n");
        compute_histogram(data, width, height, stride, hist); // Fallback to standard histogram
        return;
    }

    for (int i = 0; i < num_threads; i++) {
        private_hists[i] = calloc(256, sizeof(int));
        if (!private_hists[i]) {
            av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory for private histogram %d.\n", i);
            // Free previously allocated histograms
            for (int j = 0; j < i; j++) {
                free(private_hists[j]);
            }
            free(private_hists);
            compute_histogram(data, width, height, stride, hist); // Fallback
            return;
        }
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int *local_hist = private_hists[thread_id];

        #pragma omp for nowait
        for (int y = 0; y < height; y++) {
            const uint8_t *row = data + y * stride;
            for (int x = 0; x < width; x++) {
                local_hist[row[x]]++;
            }
        }
    }

    // Aggregate private histograms into the shared histogram
    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < 256; j++) {
            hist[j] += private_hists[i][j];
        }
        free(private_hists[i]);
    }
    free(private_hists);
#else
    compute_histogram(data, width, height, stride, hist);
#endif
}

/**
 * @brief Function to compute Histogram Difference (Chi-Square)
 */
static double compute_hist_diff(int *hist1, int *hist2) {
    double chi_sq = 0.0;
    for (int i = 0; i < 256; i++) {
        double numerator = (double)(hist1[i] - hist2[i]) * (double)(hist1[i] - hist2[i]);
        double denominator = (double)(hist1[i] + hist2[i] + 1e-6); // Avoid division by zero
        chi_sq += numerator / denominator;
    }
    return chi_sq;
}

/**
 * @brief Function to compute SSIM
 */
static double compute_SSIM(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride) {
    // Constants for SSIM
    const double C1 = 6.5025, C2 = 58.5225;

    double mean_prev = 0.0, mean_curr = 0.0;
    double variance_prev = 0.0, variance_curr = 0.0, covariance = 0.0;

    // First pass: Compute means
    #pragma omp parallel for reduction(+:mean_prev, mean_curr)
    for (int y = 0; y < height; y++) {
        const uint8_t *prev_row = prev + y * stride;
        const uint8_t *curr_row = curr + y * stride;
        for (int x = 0; x < width; x++) {
            mean_prev += prev_row[x];
            mean_curr += curr_row[x];
        }
    }
    mean_prev /= (width * height);
    mean_curr /= (width * height);

    // Second pass: Compute variances and covariance
    #pragma omp parallel for reduction(+:variance_prev, variance_curr, covariance)
    for (int y = 0; y < height; y++) {
        const uint8_t *prev_row = prev + y * stride;
        const uint8_t *curr_row = curr + y * stride;
        for (int x = 0; x < width; x++) {
            double diff_prev = prev_row[x] - mean_prev;
            double diff_curr = curr_row[x] - mean_curr;
            variance_prev += diff_prev * diff_prev;
            variance_curr += diff_curr * diff_curr;
            covariance += diff_prev * diff_curr;
        }
    }
    variance_prev /= (width * height - 1);
    variance_curr /= (width * height - 1);
    covariance /= (width * height - 1);

    // Compute SSIM
    double numerator = (2 * mean_prev * mean_curr + C1) * (2 * covariance + C2);
    double denominator = (mean_prev * mean_prev + mean_curr * mean_curr + C1) * (variance_prev + variance_curr + C2);

    return numerator / denominator;
}

/**
 * @brief Optimized and Multi-Threaded SSIM Computation Using NEON Intrinsics
 * 
 * @param prev_data Pointer to the previous grayscale frame data
 * @param curr_data Pointer to the current grayscale frame data
 * @param width Width of the frame
 * @param height Height of the frame
 * @param stride Stride of the frame data
 * @return double SSIM value
 */
static double compute_SSIM_NEON(const uint8_t *prev_data, const uint8_t *curr_data, int width, int height, int stride) {
#ifdef __ARM_NEON
    double ssim_total = 0.0;

    // Define window size (e.g., 8x8)
    const int window_size = 8;
    const int num_windows_x = width / window_size;
    const int num_windows_y = height / window_size;

    #pragma omp parallel for reduction(+:ssim_total) schedule(dynamic)
    for (int y = 0; y < num_windows_y; y++) {
        for (int x = 0; x < num_windows_x; x++) {
            const uint8_t *prev_window = prev_data + y * window_size * stride + x * window_size;
            const uint8_t *curr_window = curr_data + y * window_size * stride + x * window_size;

            uint8x8_t prev_vec = vld1_u8(prev_window);
            uint8x8_t curr_vec = vld1_u8(curr_window);

            uint16x8_t prev_extended = vmovl_u8(prev_vec);
            uint16x8_t curr_extended = vmovl_u8(curr_vec);

            // Sum using vaddvq_u16 for the full vector
            double mean_prev = (double)vaddvq_u16(prev_extended) / (window_size * window_size);
            double mean_curr = (double)vaddvq_u16(curr_extended) / (window_size * window_size);

            double variance_prev = 0.0, variance_curr = 0.0, covariance = 0.0;

            for (int i = 0; i < window_size * window_size; i += 8) {
                uint8x8_t p = vld1_u8(prev_window + i);
                uint8x8_t c = vld1_u8(curr_window + i);

                uint16x8_t p16 = vmovl_u8(p);
                uint16x8_t c16 = vmovl_u8(c);

                float32x4_t p_f_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(p16)));
                float32x4_t p_f_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(p16)));
                float32x4_t c_f_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
                float32x4_t c_f_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));

                float32x4_t diff_p_low = vsubq_f32(p_f_low, vdupq_n_f32((float)mean_prev));
                float32x4_t diff_p_high = vsubq_f32(p_f_high, vdupq_n_f32((float)mean_prev));
                float32x4_t diff_c_low = vsubq_f32(c_f_low, vdupq_n_f32((float)mean_curr));
                float32x4_t diff_c_high = vsubq_f32(c_f_high, vdupq_n_f32((float)mean_curr));

                float32x4_t var_p_low = vmulq_f32(diff_p_low, diff_p_low);
                float32x4_t var_p_high = vmulq_f32(diff_p_high, diff_p_high);
                float32x4_t var_c_low = vmulq_f32(diff_c_low, diff_c_low);
                float32x4_t var_c_high = vmulq_f32(diff_c_high, diff_c_high);

                float32x4_t cov_low = vmulq_f32(diff_p_low, diff_c_low);
                float32x4_t cov_high = vmulq_f32(diff_p_high, diff_c_high);

                variance_prev += (double)(vaddvq_f32(var_p_low) + vaddvq_f32(var_p_high));
                variance_curr += (double)(vaddvq_f32(var_c_low) + vaddvq_f32(var_c_high));
                covariance += (double)(vaddvq_f32(cov_low) + vaddvq_f32(cov_high));
            }

            variance_prev /= (window_size * window_size - 1);
            variance_curr /= (window_size * window_size - 1);
            covariance /= (window_size * window_size - 1);

            double C1 = 6.5025, C2 = 58.5225;
            double ssim_window = ((2 * mean_prev * mean_curr + C1) * (2 * covariance + C2)) /
                                  ((mean_prev * mean_prev + mean_curr * mean_curr + C1) * (variance_prev + variance_curr + C2));
            ssim_total += ssim_window;
        }
    }

    return ssim_total / (num_windows_x * num_windows_y);
#else
    return compute_SSIM(prev_data, curr_data, width, height, stride);
#endif
}

/**
 * @brief Function to compute SAD
 */
static double compute_SAD(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride) {
    double sad = 0.0;
    #pragma omp parallel for reduction(+:sad)
    for (int y = 0; y < height; y++) {
        const uint8_t *p = prev + y * stride;
        const uint8_t *c = curr + y * stride;
        for (int x = 0; x < width; x++) {
            sad += fabs((double)p[x] - (double)c[x]);
        }
    }
    return sad;
}

/**
 * @brief Function to compute SAD using NEON intrinsics
 */
static double compute_SAD_NEON(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride) {
#ifdef __ARM_NEON
    uint64_t sad = 0;

    // Iterate over blocks
    for (int y = 0; y < height; y += BLOCK_SIZE) {
        for (int x = 0; x < width; x += BLOCK_SIZE) {

            // Process each row within the block
            for (int by = 0; by < BLOCK_SIZE && (y + by) < height; by++) {
                const uint8_t *prev_row = prev + (y + by) * stride + x;
                const uint8_t *curr_row = curr + (y + by) * stride + x;

                // Prefetch the rows for caching
                __builtin_prefetch(prev_row, 0, 3);
                __builtin_prefetch(curr_row, 0, 3);

                int bx = 0;
                for (; bx <= BLOCK_SIZE - 16 && (x + bx) < width; bx += 16) {
                    uint8x16_t prev_vals = vld1q_u8(prev_row + bx);
                    uint8x16_t curr_vals = vld1q_u8(curr_row + bx);

                    // Calculate absolute differences and accumulate SAD
                    uint8x16_t abs_diff = vabdq_u8(prev_vals, curr_vals);
                    sad += vaddvq_u16(vpaddlq_u8(abs_diff));
                }

                // Process any remaining pixels in the row
                for (; bx < BLOCK_SIZE && (x + bx) < width; bx++) {
                    sad += abs((int32_t)prev_row[bx] - (int32_t)curr_row[bx]);
                }
            }
        }
    }
    return (double)sad;
#else
    return compute_SAD(prev, curr, width, height, stride);
#endif
}

/**
 * @brief Function to compute DCT Energy
 */
static double compute_DCT_energy(CaeContext *s, const uint8_t *data, int width, int height, int stride) {
    // Populate input with normalized pixel values
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const uint8_t *row = data + y * stride;
        for (int x = 0; x < width; x++) {
            s->dct_input[y * width + x] = (double)row[x] / 255.0;
        }
    }

    // Execute DCT
    fftw_execute(s->dct_plan);

    // Compute energy
    double energy = 0.0;
    int fft_size = height * (width / 2 + 1); // FFTW R2C output size
    #pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < fft_size; i++) {
        double real = s->dct_output[i][0];
        double imag = s->dct_output[i][1];
        energy += real * real + imag * imag;
    }

    return energy;
}

/**
 * @brief Optimized Sobel Filter Using NEON Intrinsics and OpenMP
 * 
 * This function computes the total Sobel energy (sum of gradient magnitudes) 
 * for a given grayscale image using ARM NEON vectorization and OpenMP parallelization.
 * It processes 16 pixels per iteration to maximize SIMD throughput.
 * 
 * @param src Pointer to the input grayscale image data (16-byte aligned)
 * @param width Width of the image (must be >=3)
 * @param height Height of the image (must be >=3)
 * @param stride Stride (bytes per row) of the input image
 * @return double Total Sobel energy for the frame
 */
static double compute_Sobel_energy_NEON(const uint8_t *restrict src, int width, int height, int stride) {
#ifdef __ARM_NEON
    if (width < 3 || height < 3) return 0.0;

    // Calculate SIMD processing width (16 pixels per iteration)
    int simd_width = (width - 2) / 16 * 16;
    double total_sobel_energy = 0.0;

    // Define constants for Sobel
    const int8x16_t neg2 = vdupq_n_s8(-2);
    const int8x16_t pos2 = vdupq_n_s8(2);
    const int8x16_t neg1 = vdupq_n_s8(-1);
    const int8x16_t pos1 = vdupq_n_s8(1);

    #pragma omp parallel for reduction(+:total_sobel_energy) schedule(static)
    for (int y = 1; y < height - 1; y++) {
        const uint8_t *prev_row = src + (y - 1) * stride;
        const uint8_t *curr_row = src + y * stride;
        const uint8_t *next_row = src + (y + 1) * stride;

        double row_sobel_energy = 0.0;

        for (int x = 1; x <= simd_width; x += 16) {
            // Load 16 pixels from each relevant position
            uint8x16_t prev_left_u8   = vld1q_u8(prev_row + x - 1);
            uint8x16_t prev_right_u8  = vld1q_u8(prev_row + x + 1);
            uint8x16_t curr_left_u8   = vld1q_u8(curr_row + x - 1);
            uint8x16_t curr_right_u8  = vld1q_u8(curr_row + x + 1);
            uint8x16_t next_left_u8   = vld1q_u8(next_row + x - 1);
            uint8x16_t next_right_u8  = vld1q_u8(next_row + x + 1);
            uint8x16_t prev_center_u8 = vld1q_u8(prev_row + x);
            uint8x16_t next_center_u8 = vld1q_u8(next_row + x);

            // Reinterpret as signed integers
            int8x16_t p_prev_left    = vreinterpretq_s8_u8(prev_left_u8);
            int8x16_t p_prev_right   = vreinterpretq_s8_u8(prev_right_u8);
            int8x16_t p_curr_left    = vreinterpretq_s8_u8(curr_left_u8);
            int8x16_t p_curr_right   = vreinterpretq_s8_u8(curr_right_u8);
            int8x16_t p_next_left    = vreinterpretq_s8_u8(next_left_u8);
            int8x16_t p_next_right   = vreinterpretq_s8_u8(next_right_u8);
            int8x16_t p_prev_center  = vreinterpretq_s8_u8(prev_center_u8);
            int8x16_t p_next_center  = vreinterpretq_s8_u8(next_center_u8);

            // Initialize gx and gy vectors
            int16x8_t gx_low = vmull_s8(vget_low_s8(p_prev_left), vget_low_s8(neg1));
            gx_low = vmlal_s8(gx_low, vget_low_s8(p_curr_left), vget_low_s8(neg2));
            gx_low = vmlal_s8(gx_low, vget_low_s8(p_next_left), vget_low_s8(neg1));
            gx_low = vmlal_s8(gx_low, vget_low_s8(p_prev_right), vget_low_s8(pos1));
            gx_low = vmlal_s8(gx_low, vget_low_s8(p_curr_right), vget_low_s8(pos2));
            gx_low = vmlal_s8(gx_low, vget_low_s8(p_next_right), vget_low_s8(pos1));

            int16x8_t gx_high = vmull_s8(vget_high_s8(p_prev_left), vget_high_s8(neg1));
            gx_high = vmlal_s8(gx_high, vget_high_s8(p_curr_left), vget_high_s8(neg2));
            gx_high = vmlal_s8(gx_high, vget_high_s8(p_next_left), vget_high_s8(neg1));
            gx_high = vmlal_s8(gx_high, vget_high_s8(p_prev_right), vget_high_s8(pos1));
            gx_high = vmlal_s8(gx_high, vget_high_s8(p_curr_right), vget_high_s8(pos2));
            gx_high = vmlal_s8(gx_high, vget_high_s8(p_next_right), vget_high_s8(pos1));

            // Calculate gy
            int16x8_t gy_low = vmull_s8(vget_low_s8(p_prev_left), vget_low_s8(pos1));
            gy_low = vmlal_s8(gy_low, vget_low_s8(p_prev_center), vdup_n_s8(2)); // p_prev_center * 2
            gy_low = vmlal_s8(gy_low, vget_low_s8(p_prev_right), vget_low_s8(pos1));
            gy_low = vmlal_s8(gy_low, vget_low_s8(p_next_left), vget_low_s8(neg1));
            gy_low = vmlal_s8(gy_low, vget_low_s8(p_next_center), vdup_n_s8(-2)); // p_next_center * -2
            gy_low = vmlal_s8(gy_low, vget_low_s8(p_next_right), vget_low_s8(neg1));

            int16x8_t gy_high = vmull_s8(vget_high_s8(p_prev_left), vget_high_s8(pos1));
            gy_high = vmlal_s8(gy_high, vget_high_s8(p_prev_center), vdup_n_s8(2)); // p_prev_center * 2
            gy_high = vmlal_s8(gy_high, vget_high_s8(p_prev_right), vget_high_s8(pos1));
            gy_high = vmlal_s8(gy_high, vget_high_s8(p_next_left), vget_high_s8(neg1));
            gy_high = vmlal_s8(gy_high, vget_high_s8(p_next_center), vdup_n_s8(-2)); // p_next_center * -2
            gy_high = vmlal_s8(gy_high, vget_high_s8(p_next_right), vget_high_s8(neg1));

            // Sum of absolute values of gx and gy
            uint16x8_t abs_gx_low = vqabsq_s16(gx_low);
            uint16x8_t abs_gy_low = vqabsq_s16(gy_low);
            uint16x8_t sum_gx_gy_low = vaddq_u16(abs_gx_low, abs_gy_low);

            uint16x8_t abs_gx_high = vqabsq_s16(gx_high);
            uint16x8_t abs_gy_high = vqabsq_s16(gy_high);
            uint16x8_t sum_gx_gy_high = vaddq_u16(abs_gx_high, abs_gy_high);

            // Accumulate the sum
            // Convert to unsigned integers and then to integers for accumulation
            // vaddvq_u16 sums all elements in the vector
            uint32_t sum_low = vaddvq_u16(sum_gx_gy_low);
            uint32_t sum_high = vaddvq_u16(sum_gx_gy_high);
            row_sobel_energy += (double)(sum_low + sum_high);
        }

        // Handle any remaining pixels not processed by SIMD
        for (int x = simd_width + 1; x < width - 1; x++) {
            int gx = -1 * prev_row[x - 1] + 1 * prev_row[x + 1]
                   -2 * curr_row[x - 1] + 2 * curr_row[x + 1]
                   -1 * next_row[x - 1] + 1 * next_row[x + 1];
            int gy = -1 * prev_row[x - 1] - 2 * prev_row[x] -1 * prev_row[x + 1]
                   +1 * next_row[x - 1] + 2 * next_row[x] +1 * next_row[x + 1];
            // Use |gx| + |gy| as an approximation for magnitude to avoid sqrt
            double magnitude = fabs((double)gx) + fabs((double)gy);
            row_sobel_energy += magnitude;
        }

        // Accumulate row_sobel_energy to total_sobel_energy
        total_sobel_energy += row_sobel_energy;
    }

    return total_sobel_energy / ((double)(width - 2) * (double)(height - 2));
#else
    return compute_Sobel_energy(src, width, height, stride);
#endif
}

/**
 * @brief Function to compute Sobel Energy
 */
static double compute_Sobel_energy(const uint8_t *data, int width, int height, int stride) {
    double sobel_energy = 0.0;

    // Define Sobel kernels
    const int sobel_x[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };
    const int sobel_y[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    // Compute gradients
    #pragma omp parallel for reduction(+:sobel_energy)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0;
            int gy = 0;
            for (int ky = -1; ky <=1; ky++) {
                for (int kx = -1; kx <=1; kx++) {
                    gx += sobel_x[ky + 1][kx + 1] * data[(y + ky) * stride + (x + kx)];
                    gy += sobel_y[ky + 1][kx + 1] * data[(y + ky) * stride + (x + kx)];
                }
            }
            double magnitude = sqrt((double)(gx * gx + gy * gy));
            sobel_energy += magnitude;
        }
    }

    return sobel_energy;
}

/**
 * @brief Function to compute Entropy
 */
static double compute_entropy(const uint8_t *data, int width, int height, int stride) {
    int hist[256] = {0};
    compute_histogram_NEON(data, width, height, stride, hist);

    double entropy = 0.0;
    double total = (double)(width * height);

    for (int i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            double p = hist[i] / total;
            entropy -= p * log(p);
        }
    }

    return entropy;
}

/**
 * @brief Optimized Entropy Computation Using NEON Intrinsics and OpenMP
 * 
 * This function computes the entropy of a grayscale image using NEON vectorization and OpenMP parallelization.
 * 
 * @param data Pointer to the input grayscale image data
 * @param width Width of the image
 * @param height Height of the image
 * @param stride Stride (bytes per row) of the input image
 * @return double Entropy of the image
 */
static double compute_entropy_NEON(const uint8_t *data, int width, int height, int stride) {
#ifdef __ARM_NEON
    // Initialize a histogram array
    uint32_t histogram[256] = {0};
    double entropy = 0.0;

    // Parallelize the loop over blocks using OpenMP
    #pragma omp parallel
    {
        // Each thread maintains its own local histogram to avoid race conditions
        uint32_t local_histogram[256] = {0};

        #pragma omp for nowait schedule(static)
        for (int y = 0; y < height; y += BLOCK_SIZE) {
            for (int x = 0; x < width; x += BLOCK_SIZE) {

                // Iterate through each row in the block
                for (int by = 0; by < BLOCK_SIZE && (y + by) < height; by++) {
                    const uint8_t *row = data + (y + by) * stride + x;

                    // Prefetch row data to improve cache performance
                    __builtin_prefetch(row, 0, 3);

                    int bx = 0;

                    // Process 8 pixels at a time
                    for (; bx <= BLOCK_SIZE - 8 && (x + bx + 7) < width; bx += 8) {
                        // Load 8 pixels
                        uint8x8_t pixels = vld1_u8(row + bx);

                        // Convert to uint16x8_t
                        uint16x8_t pixels_16 = vmovl_u8(pixels); // Zero-extend to 16 bits

                        // Split into lower and higher 4 pixels each
                        uint16x4_t pixels_low = vget_low_u16(pixels_16);
                        uint16x4_t pixels_high = vget_high_u16(pixels_16);

                        // Convert to uint32x4_t
                        uint32x4_t pixels_low32 = vmovl_u16(pixels_low);
                        uint32x4_t pixels_high32 = vmovl_u16(pixels_high);

                        // Convert to float32x4_t
                        float32x4_t p1 = vcvtq_f32_u32(pixels_low32);
                        float32x4_t p2 = vcvtq_f32_u32(pixels_high32);

                        // Accumulate histogram bins
                        for (int i = 0; i < 4; i++) {
                            uint8_t pixel = (uint8_t)p1[i];
                            local_histogram[pixel]++;
                        }
                        for (int i = 0; i < 4; i++) {
                            uint8_t pixel = (uint8_t)p2[i];
                            local_histogram[pixel]++;
                        }
                    }

                    // Handle any remaining pixels
                    for (; bx < BLOCK_SIZE && (x + bx) < width; bx++) {
                        uint8_t pixel = row[bx];
                        local_histogram[pixel]++;
                    }
                }
            }
        }

        // Combine local histograms into the global histogram
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++) {
                histogram[i] += local_histogram[i];
            }
        }
    } // End of parallel region

    // Compute entropy from the global histogram
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            double p = (double)histogram[i] / (double)(width * height);
            entropy -= p * log2(p);
        }
    }

    return entropy;
#else
    return compute_entropy(data, width, height, stride);
#endif
}

/**
 * @brief Function to compute Color Variance
 * 
 * Assumes grayscale for simplicity. For color images, handle each channel separately.
 */
static double compute_color_variance(const uint8_t *data, int width, int height, int stride) {
    double mean = 0.0;
    double variance = 0.0;

    // First pass: compute mean
    #pragma omp parallel for reduction(+:mean)
    for (int y = 0; y < height; y++) {
        const uint8_t *row = data + y * stride;
        for (int x = 0; x < width; x++) {
            mean += row[x];
        }
    }
    mean /= (width * height);

    // Second pass: compute variance
    #pragma omp parallel for reduction(+:variance)
    for (int y = 0; y < height; y++) {
        const uint8_t *row = data + y * stride;
        for (int x = 0; x < width; x++) {
            double diff = row[x] - mean;
            variance += diff * diff;
        }
    }
    variance /= (width * height - 1);

    return variance;
}

/**
 * @brief Compute color variance in BLOCK_SIZE x BLOCK_SIZE blocks using NEON
 *
 * @param data Pointer to the input image data
 * @param width Width of the image
 * @param height Height of the image
 * @param stride Stride of the image (number of bytes per row)
 * @return double The total color variance for the entire image
 */
static double compute_color_variance_NEON(const uint8_t *data, int width, int height, int stride) {
#ifdef __ARM_NEON
    double sum = 0.0;
    double sum_sq = 0.0;

    // Parallelize the loop over blocks using OpenMP
    #pragma omp parallel for reduction(+:sum, sum_sq) schedule(static)
    for (int y = 0; y < height; y += BLOCK_SIZE) {
        for (int x = 0; x < width; x += BLOCK_SIZE) {

            // Iterate through each row in the block
            for (int by = 0; by < BLOCK_SIZE && (y + by) < height; by++) {
                const uint8_t *row = data + (y + by) * stride + x;

                // Prefetch row data to improve cache performance
                __builtin_prefetch(row, 0, 3);

                int bx = 0;

                // Initialize NEON vectors for sum and sum of squares
                float32x4_t v_sum = vdupq_n_f32(0.0f);
                float32x4_t v_sum_sq = vdupq_n_f32(0.0f);

                // Process 8 pixels at a time
                for (; bx <= BLOCK_SIZE - 8 && (x + bx + 7) < width; bx += 8) {
                    // Load 8 pixels
                    uint8x8_t pixels = vld1_u8(row + bx);

                    // Convert to uint16x8_t
                    uint16x8_t pixels_16 = vmovl_u8(pixels); // Zero-extend to 16 bits

                    // Split into lower and higher 4 pixels each
                    uint16x4_t pixels_low = vget_low_u16(pixels_16);
                    uint16x4_t pixels_high = vget_high_u16(pixels_16);

                    // Convert to uint32x4_t
                    uint32x4_t pixels_low32 = vmovl_u16(pixels_low);
                    uint32x4_t pixels_high32 = vmovl_u16(pixels_high);

                    // Convert to float32x4_t
                    float32x4_t p1 = vcvtq_f32_u32(pixels_low32);
                    float32x4_t p2 = vcvtq_f32_u32(pixels_high32);

                    // Accumulate sum
                    v_sum = vaddq_f32(v_sum, vaddq_f32(p1, p2));

                    // Accumulate sum of squares
                    float32x4_t p1_sq = vmulq_f32(p1, p1);
                    float32x4_t p2_sq = vmulq_f32(p2, p2);
                    v_sum_sq = vaddq_f32(v_sum_sq, vaddq_f32(p1_sq, p2_sq));
                }

                // Horizontal add to accumulate NEON vectors
                sum += vaddvq_f32(v_sum);     // Sum all elements in v_sum
                sum_sq += vaddvq_f32(v_sum_sq); // Sum all elements in v_sum_sq

                // Handle any remaining pixels
                for (; bx < BLOCK_SIZE && (x + bx) < width; bx++) {
                    float pixel = (float)row[bx];
                    sum += pixel;
                    sum_sq += pixel * pixel;
                }
            }
        }
    }

    // Compute mean and variance
    double total_pixels = (double)(width * height);
    double mean = sum / total_pixels;
    double variance = (sum_sq / total_pixels) - (mean * mean);

    return variance;
#else
    return compute_color_variance(data, width, height, stride);
#endif
}

/**
 * @brief Function to adjust weights dynamically based on recent frame impacts
 */
static void adjust_weights(CaeContext *s, double delta_complexity, double delta_ssim, double delta_hist, double delta_dct, double delta_sobel, double delta_entropy, double delta_color_var) {
    // Adjust weights based on whether each metric exceeds its threshold
    if (delta_complexity > s->alpha_complexity)
        s->weight_complexity += 0.1;
    else
        s->weight_complexity = fmax(s->weight_complexity - 0.05, 0.1);

    if (delta_ssim > s->alpha_ssim)
        s->weight_ssim += 0.1;
    else
        s->weight_ssim = fmax(s->weight_ssim - 0.05, 0.1);

    if (delta_hist > s->alpha_hist)
        s->weight_hist += 0.1;
    else
        s->weight_hist = fmax(s->weight_hist - 0.05, 0.1);

    if (delta_dct > s->alpha_dct)
        s->weight_dct += 0.1;
    else
        s->weight_dct = fmax(s->weight_dct - 0.05, 0.1);

    if (delta_sobel > s->alpha_sobel)
        s->weight_sobel += 0.1;
    else
        s->weight_sobel = fmax(s->weight_sobel - 0.05, 0.1);

    if (delta_entropy > 0.5) // Example threshold for entropy
        s->weight_entropy += 0.1;
    else
        s->weight_entropy = fmax(s->weight_entropy - 0.05, 0.1);

    if (delta_color_var > 100.0) // Example threshold for color variance
        s->weight_color_var += 0.1;
    else
        s->weight_color_var = fmax(s->weight_color_var - 0.05, 0.1);

    // Normalize weights to sum to max_weight (e.g., 10.0 to prevent domination)
    double total_weight = s->weight_complexity + s->weight_ssim + s->weight_hist + s->weight_dct + s->weight_sobel + s->weight_entropy + s->weight_color_var;
    if (total_weight > s->max_weight) {
        double scaling_factor = s->max_weight / total_weight;
        s->weight_complexity *= scaling_factor;
        s->weight_ssim *= scaling_factor;
        s->weight_hist *= scaling_factor;
        s->weight_dct *= scaling_factor;
        s->weight_sobel *= scaling_factor;
        s->weight_entropy *= scaling_factor;
        s->weight_color_var *= scaling_factor;
    }
}

/**
 * @brief Function to calculate adaptive threshold based on weighted_sum_window
 */
static bool calculate_adaptive_threshold(CaeContext *s) {
    if (!s->weighted_sum_window || s->weighted_sum_window_size <= 0)
        return false;

    // Calculate median and MAD
    bool success = calculate_median(s->weighted_sum_window, s->weighted_sum_window_size, &s->median_weighted_sum);
    if (!success)
        return false;

    success = calculate_mad(s->weighted_sum_window, s->weighted_sum_window_size, s->median_weighted_sum, &s->mad_weighted_sum);
    if (!success)
        return false;

    return true;

    // Set Adaptive_Threshold
    // Already handled in the detection condition
    // No action needed here unless you want to store or use the threshold elsewhere
}

/**
 * @brief Configure the properties of the output link to match the input.
 *
 * @param inlink The input link.
 * @return int 0 on success, negative AVERROR on failure.
 */
static int cae_config_props(AVFilterLink *inlink) {
    AVFilterContext *ctx = inlink->dst;
    CaeContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    // Set output link properties to match input
    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    outlink->format = inlink->format; // Corrected Line
    outlink->time_base = inlink->time_base;

    s->src_pix_fmt = inlink->format;
    s->width = inlink->w;
    s->height = inlink->h;

    // Initialize SwsContext for color space conversion (e.g., YUV420P to GRAY)
    s->sws_ctx = sws_getContext(
        s->width,
        s->height,
        s->src_pix_fmt,
        s->width,
        s->height,
        AV_PIX_FMT_GRAY8,
        SWS_BILINEAR,
        NULL,
        NULL,
        NULL
    );

    if (!s->sws_ctx) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize SwsContext for color space conversion.\n");
        return AVERROR(EINVAL);
    }

    s->dst_pix_fmt = AV_PIX_FMT_GRAY8;

    // Allocate and initialize the previous grayscale frame
    s->prev_gray_frame = ff_get_video_buffer(ctx->outputs[0], s->width, s->height);
    if (!s->prev_gray_frame) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate AVFrame for previous grayscale data.\n");
        sws_freeContext(s->sws_ctx);
        return AVERROR(ENOMEM);
    }

    // Initialize previous grayscale frame to zero
    if (sws_scale(
            s->sws_ctx,
            (const uint8_t * const*)s->prev_gray_frame->data,
            s->prev_gray_frame->linesize,
            0,
            s->height,
            s->prev_gray_frame->data,
            s->prev_gray_frame->linesize) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize previous grayscale frame.\n");
        av_frame_free(&s->prev_gray_frame);
        sws_freeContext(s->sws_ctx);
        return AVERROR(EINVAL);
    }

    // Initialize FFTW multithreading
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads()); // Set FFTW to use maximum available threads

    // Initialize FFTW plan for DCT
    s->dct_input = fftw_alloc_real(s->width * s->height);
    s->dct_output = fftw_alloc_complex(s->width * (s->height / 2 + 1));
    if (!s->dct_input || !s->dct_output) {
        av_log(ctx, AV_LOG_ERROR, "FFTW allocation failed for DCT.\n");
        if (s->dct_input) fftw_free(s->dct_input);
        if (s->dct_output) fftw_free(s->dct_output);
        av_frame_free(&s->prev_gray_frame);
        sws_freeContext(s->sws_ctx);
        return AVERROR(ENOMEM);
    }

    s->dct_plan = fftw_plan_dft_r2c_2d(s->height, s->width, s->dct_input, s->dct_output, FFTW_MEASURE);
    if (!s->dct_plan) {
        av_log(ctx, AV_LOG_ERROR, "FFTW plan creation failed for DCT.\n");
        fftw_free(s->dct_input);
        fftw_free(s->dct_output);
        av_frame_free(&s->prev_gray_frame);
        sws_freeContext(s->sws_ctx);
        return AVERROR(EINVAL);
    }

    return 0;
}

/**
 * @brief Initialize the CAE context with default or configured values.
 *
 * @param ctx Pointer to the CAE context.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static int init_cae_context(AVFilterContext *ctx) {
    CaeContext  *s = ctx->priv;

    // Initialize default values if not set
    if (s->alpha_complexity <= 0.0)
        s->alpha_complexity = 0.5;
    if (s->alpha_ssim <= 0.0)
        s->alpha_ssim = 0.8;
    if (s->alpha_hist <= 0.0)
        s->alpha_hist = 0.5;
    if (s->alpha_dct <= 0.0)
        s->alpha_dct = 0.5;
    if (s->alpha_sobel <= 0.0)
        s->alpha_sobel = 0.5;
    if (s->window_size <= 0)
        s->window_size = 10;
    if (s->threshold_mode < 0)
        s->threshold_mode = 0;
    if (s->cooldown_frames < 0)
        s->cooldown_frames = 10; // Default cooldown
    if (s->required_consecutive_changes < 1)
        s->required_consecutive_changes = 2; // Default consecutive detections
    if (s->k_threshold <= 0.0)
        s->k_threshold = 3.0; // Default value
    if (s->max_weight <= 0.0)
        s->max_weight = 10.0; // Prevent weight domination

    // Initialize adaptive frame interval limits
    if (s->min_frame_interval <= 0)
        s->min_frame_interval = 1;
    if (s->max_frame_interval <= s->min_frame_interval)
        s->max_frame_interval = s->min_frame_interval + 5; // Example: min + 5 = 6
    
    // Initialize new normalization variables
    s->min_complexity = DBL_MAX;
    s->max_complexity = -DBL_MAX;
    s->min_ssim = DBL_MAX;
    s->max_ssim = -DBL_MAX;
    s->min_hist = DBL_MAX;
    s->max_hist = -DBL_MAX;
    s->min_dct = DBL_MAX;
    s->max_dct = -DBL_MAX;
    s->min_sobel = DBL_MAX;
    s->max_sobel = -DBL_MAX;
    s->min_entropy = DBL_MAX;
    s->max_entropy = -DBL_MAX;
    s->min_color_var = DBL_MAX;
    s->max_color_var = -DBL_MAX;

    // Initialize crf_exponent
    if (s->crf_exponent <= 0.0)
        s->crf_exponent = 2.0; // Default exponent value

    // Store the filter context for logging
    s->ctx = ctx;


    // Allocate memory for the sliding windows
    s->complexity_window = (double*)av_malloc(s->window_size * sizeof(double));
    s->ssim_window = (double*)av_malloc(s->window_size * sizeof(double));
    s->hist_window = (double*)av_malloc(s->window_size * sizeof(double));
    s->dct_window = (double*)av_malloc(s->window_size * sizeof(double));
    s->sobel_window = (double*)av_malloc(s->window_size * sizeof(double));
    s->entropy_window = (double*)av_malloc(s->window_size * sizeof(double));
    s->color_var_window = (double*)av_malloc(s->window_size * sizeof(double));
    if (!s->complexity_window || !s->ssim_window || !s->hist_window ||
        !s->dct_window || !s->sobel_window || !s->entropy_window ||
        !s->color_var_window) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for sliding windows.\n");
        av_freep(&s->complexity_window);
        av_freep(&s->ssim_window);
        av_freep(&s->hist_window);
        av_freep(&s->dct_window);
        av_freep(&s->sobel_window);
        av_freep(&s->entropy_window);
        av_freep(&s->color_var_window);
        return AVERROR(ENOMEM);
    }
    memset(s->complexity_window, 0, s->window_size * sizeof(double));
    memset(s->ssim_window, 0, s->window_size * sizeof(double));
    memset(s->hist_window, 0, s->window_size * sizeof(double));
    memset(s->dct_window, 0, s->window_size * sizeof(double));
    memset(s->sobel_window, 0, s->window_size * sizeof(double));
    memset(s->entropy_window, 0, s->window_size * sizeof(double));
    memset(s->color_var_window, 0, s->window_size * sizeof(double));

    s->window_index = 0;
    s->window_filled = false;
    s->previous_complexity = 0.0;

    // Initialize scene change detection enhancements
    s->current_cooldown = 0;
    s->consecutive_detected = 0;

    // Initialize dynamic weights
    s->weight_complexity = 1.0;    // Balanced importance
    s->weight_ssim = 1.2;           // High importance due to structural similarity
    s->weight_hist = 0.8;           // Moderate importance
    s->weight_dct = 1.5;            // High importance for frequency domain analysis
    s->weight_sobel = 1.0;          // Balanced for edge detection
    s->weight_entropy = 0.6;        // Moderate importance
    s->weight_color_var = 0.7;      // Slightly increased for color dynamics

    // Initialize weighted_sum_window
    s->weighted_sum_window_size = s->window_size;
    s->weighted_sum_window = (double*)av_malloc(s->weighted_sum_window_size * sizeof(double));
    if (!s->weighted_sum_window) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for weighted_sum_window.\n");
        av_freep(&s->complexity_window);
        av_freep(&s->ssim_window);
        av_freep(&s->hist_window);
        av_freep(&s->dct_window);
        av_freep(&s->sobel_window);
        av_freep(&s->entropy_window);
        av_freep(&s->color_var_window);
        return AVERROR(ENOMEM);
    }
    memset(s->weighted_sum_window, 0, s->weighted_sum_window_size * sizeof(double));
    s->weighted_sum_window_index = 0;
    s->weighted_sum_window_filled = false;

    return 0;
}

/**
 * @brief Uninitialize the CAE context by freeing allocated resources.
 *
 * @param ctx Pointer to the CAE context.
 */
static void uninit_cae_context(AVFilterContext *ctx) {
    CaeContext *s = ctx->priv;

    if (s->complexity_window)
        av_free(s->complexity_window);
    if (s->ssim_window)
        av_free(s->ssim_window);
    if (s->hist_window)
        av_free(s->hist_window);
    if (s->dct_window)
        av_free(s->dct_window);
    if (s->sobel_window)
        av_free(s->sobel_window);
    if (s->entropy_window)
        av_free(s->entropy_window);
    if (s->color_var_window)
        av_free(s->color_var_window);
    if (s->weighted_sum_window)
        av_free(s->weighted_sum_window);
    if (s->prev_gray_frame)
        av_frame_free(&s->prev_gray_frame);
    if (s->sws_ctx)
        sws_freeContext(s->sws_ctx);
    if (s->dct_plan)
        fftw_destroy_plan(s->dct_plan);
    if (s->dct_input)
        fftw_free(s->dct_input);
    if (s->dct_output)
        fftw_free(s->dct_output);
    av_log(NULL, AV_LOG_INFO, "CAE context uninitialized and resources freed.\n");
}

/**
 * @brief Initialize the CAE filter.
 *
 * @param ctx The filter context.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static av_cold int init_filter(AVFilterContext *ctx) {
    int ret = init_cae_context(ctx);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize CAE context.\n");
        return ret;
    }

    av_log(ctx, AV_LOG_INFO, "CAE filter initialized successfully.\n");
    return 0;
}

/**
 * @brief Uninitialize the CAE filter.
 *
 * @param ctx The filter context.
 */
static av_cold void uninit_filter(AVFilterContext *ctx) {
    uninit_cae_context(ctx);
    // Cleanup FFTW multithreading
    fftw_cleanup_threads();

    av_log(ctx, AV_LOG_INFO, "CAE filter uninitialized successfully.\n");
}

/**
 * @brief Main Filter Frame Function
 *
 * Processes each incoming frame to detect scene changes based on various metrics.
 *
 * @param inlink The input link.
 * @param frame The input frame.
 * @return int 0 on success, negative AVERROR on failure.
 */
static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
    AVFilterContext *ctx = inlink->dst;
    CaeContext *s = ctx->priv;

    START_TIMER(GLOBAL);

    // Increment the frame counter
    s->frame_counter++;

    // Determine if the current frame should be processed
    //if (s->frame_counter % s->frame_interval != 0) {
        // Optionally, log that the frame is being skipped
    //    av_log(ctx, AV_LOG_DEBUG, "Skipping frame %lld for scene change detection.\n", frame->pts);

        // Pass the frame to the next filter without processing
    //    return ff_filter_frame(ctx->outputs[0], frame);
    //}

    // Allocate grayscale frame for internal processing
    AVFrame *gray_frame = ff_get_video_buffer(ctx->outputs[0], s->width, s->height);
    if (!gray_frame) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate grayscale frame.\n");
        return AVERROR(ENOMEM);
    }

    // Convert to grayscale
    if (sws_scale(
            s->sws_ctx,
            (const uint8_t * const*)frame->data,
            frame->linesize,
            0,
            s->height,
            gray_frame->data,
            gray_frame->linesize) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to convert frame to grayscale.\n");
        av_frame_free(&gray_frame);
        return AVERROR(EINVAL);
    }

    // Ensure the grayscale frame is writable
    if (av_frame_make_writable(gray_frame) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Grayscale frame not writable.\n");
        av_frame_free(&gray_frame);
        return AVERROR(EINVAL);
    }

    // Initialize metrics
    double current_complexity = 0.0;
    bool compute_metrics = false;

    // Check if previous frame exists
    if (s->window_filled || s->window_index > 0) {
        compute_metrics = true;
        // Compute SAD using NEON-optimized function
        START_TIMER(SAD_NEON);
        current_complexity = compute_SAD_NEON(
            s->prev_gray_frame->data[0],   // Previous grayscale frame data
            gray_frame->data[0],           // Current grayscale frame data
            s->width,
            s->height,
            gray_frame->linesize[0]
        );
        END_TIMER(SAD_NEON, "SAD_NEON");
    }

    // Validate current_complexity
    if (!is_finite_double(current_complexity)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for current_complexity: %f\n", current_complexity);
        current_complexity = 0.0; // Assign a default value or handle as needed
    }

    START_TIMER(HIST_NEON);
    // Compute Histograms
    compute_histogram_NEON(s->prev_gray_frame->data[0], s->width, s->height, gray_frame->linesize[0], s->hist_prev);
    compute_histogram_NEON(gray_frame->data[0], s->width, s->height, gray_frame->linesize[0], s->hist_curr);
    END_TIMER(HIST_NEON, "HIST_NEON");

    START_TIMER(HIST_DIFF_NEON);
    // Compute Histogram Difference
    double hist_diff = compute_hist_diff(s->hist_prev, s->hist_curr);
    END_TIMER(HIST_DIFF_NEON, "HIST_DIFF_NEON");
    if (!is_finite_double(hist_diff)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for hist_diff: %f\n", hist_diff);
        hist_diff = 0.0;
    }

    START_TIMER(SSIM_NEON);
    // Compute SSIM
    double ssim = compute_SSIM_NEON(
        s->prev_gray_frame->data[0],
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );
    END_TIMER(SSIM_NEON, "SSIM_NEON");

    if (!is_finite_double(ssim)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for SSIM: %f\n", ssim);
        ssim = 1.0; // SSIM can be 1.0 for identical frames
    }

    START_TIMER(DCT);
    // Compute DCT Energy
    double dct_energy = compute_DCT_energy(
        s,
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );
    END_TIMER(DCT, "DCT");

    if (!is_finite_double(dct_energy)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for DCT Energy: %f\n", dct_energy);
        dct_energy = 0.0;
    }

    // Compute Sobel Energy 
    START_TIMER(SOBEL);
    double sobel_energy = compute_Sobel_energy_NEON(
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );
    END_TIMER(SOBEL, "SOBEL");

    if (!is_finite_double(sobel_energy)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for Sobel Energy: %f\n", sobel_energy);
        sobel_energy = 0.0;
    }

    // Compute Entropy
    START_TIMER(ENTROPY);
    double entropy = compute_entropy_NEON(
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );
    END_TIMER(ENTROPY, "ENTROPY");

    if (!is_finite_double(entropy)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for Entropy: %f\n", entropy);
        entropy = 0.0;
    }

    // Compute Color Variance
    START_TIMER(COLOR_VARIANCE);
    double color_variance = compute_color_variance_NEON(
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );
    END_TIMER(COLOR_VARIANCE, "COLOR_VARIANCE");

    if (!is_finite_double(color_variance)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid value for Color Variance: %f\n", color_variance);
        color_variance = 0.0;
    }

    // Compute delta metrics
    double delta_complexity = fabs(current_complexity - s->previous_complexity);
    double delta_ssim = 1.0 - ssim; // SSIM ranges from 0 to 1
    double delta_hist = hist_diff;
    double delta_dct = dct_energy;       // Assuming higher DCT energy indicates more change
    double delta_sobel = sobel_energy;   // Assuming higher Sobel energy indicates more change
    double delta_entropy = entropy;      // Assuming higher entropy indicates more change
    double delta_color_var = color_variance; // Assuming higher color variance indicates more change

    // Update min and max values for normalization
    s->min_complexity = fmin(s->min_complexity, delta_complexity);
    s->max_complexity = fmax(s->max_complexity, delta_complexity);
    s->min_ssim = fmin(s->min_ssim, delta_ssim);
    s->max_ssim = fmax(s->max_ssim, delta_ssim);
    s->min_hist = fmin(s->min_hist, delta_hist);
    s->max_hist = fmax(s->max_hist, delta_hist);
    s->min_dct = fmin(s->min_dct, delta_dct);
    s->max_dct = fmax(s->max_dct, delta_dct);
    s->min_sobel = fmin(s->min_sobel, delta_sobel);
    s->max_sobel = fmax(s->max_sobel, delta_sobel);
    s->min_entropy = fmin(s->min_entropy, delta_entropy);
    s->max_entropy = fmax(s->max_entropy, delta_entropy);
    s->min_color_var = fmin(s->min_color_var, delta_color_var);
    s->max_color_var = fmax(s->max_color_var, delta_color_var);

    // Normalize metrics to 0-1 scale
    double norm_complexity = (delta_complexity - s->min_complexity) / (s->max_complexity - s->min_complexity + 1e-6);
    double norm_ssim = (delta_ssim - s->min_ssim) / (s->max_ssim - s->min_ssim + 1e-6);
    double norm_hist = (delta_hist - s->min_hist) / (s->max_hist - s->min_hist + 1e-6);
    double norm_dct = (delta_dct - s->min_dct) / (s->max_dct - s->min_dct + 1e-6);
    double norm_sobel = (delta_sobel - s->min_sobel) / (s->max_sobel - s->min_sobel + 1e-6);
    double norm_entropy = (delta_entropy - s->min_entropy) / (s->max_entropy - s->min_entropy + 1e-6);
    double norm_color_var = (delta_color_var - s->min_color_var) / (s->max_color_var - s->min_color_var + 1e-6);

    // Compute activity_score
    double total_weight = s->weight_complexity + s->weight_ssim + s->weight_hist +
                          s->weight_dct + s->weight_sobel + s->weight_entropy +
                          s->weight_color_var;

    double activity_score = (
        (norm_complexity * s->weight_complexity) +
        (norm_ssim * s->weight_ssim) +
        (norm_hist * s->weight_hist) +
        (norm_dct * s->weight_dct) +
        (norm_sobel * s->weight_sobel) +
        (norm_entropy * s->weight_entropy) +
        (norm_color_var * s->weight_color_var)
    ) / total_weight;

    // Update sliding windows with delta metrics
    s->complexity_window[s->window_index] = delta_complexity;
    s->ssim_window[s->window_index] = delta_ssim;
    s->hist_window[s->window_index] = delta_hist;
    s->dct_window[s->window_index] = delta_dct;
    s->sobel_window[s->window_index] = delta_sobel;
    s->entropy_window[s->window_index] = delta_entropy;
    s->color_var_window[s->window_index] = delta_color_var;
    s->window_index = (s->window_index + 1) % s->window_size;
    if (s->window_index == 0)
        s->window_filled = true;


    // Update weighted_sum_window
    s->weighted_sum_window[s->weighted_sum_window_index] = activity_score;
    s->weighted_sum_window_index = (s->weighted_sum_window_index + 1) % s->weighted_sum_window_size;
    if (s->weighted_sum_window_index == 0)
        s->weighted_sum_window_filled = true;

    // Caching Mechanism for Median and MAD
    // To avoid recalculating median and MAD multiple times per metric,
    // store the results in temporary variables and reuse them.

    // Initialize temporary variables for median and MAD
    double median_complexity = 0.0, mad_complexity = 0.0;
    double median_ssim = 0.0, mad_ssim = 0.0;
    double median_hist = 0.0, mad_hist = 0.0;
    double median_dct = 0.0, mad_dct = 0.0;
    double median_sobel = 0.0, mad_sobel = 0.0;
    double median_entropy = 0.0, mad_entropy = 0.0;
    double median_color_var = 0.0, mad_color_var = 0.0;

    // If window is filled and metrics should be computed
    if (compute_metrics && s->window_filled) {
        // Calculate median and MAD for each metric
        START_TIMER(MAD_MEDIAN);
        if (!calculate_median(s->complexity_window, s->window_size, &median_complexity)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for complexity.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->complexity_window, s->window_size, median_complexity, &mad_complexity)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for complexity.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        if (!calculate_median(s->ssim_window, s->window_size, &median_ssim)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for SSIM delta.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->ssim_window, s->window_size, median_ssim, &mad_ssim)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for SSIM delta.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        if (!calculate_median(s->hist_window, s->window_size, &median_hist)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for Histogram Difference.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->hist_window, s->window_size, median_hist, &mad_hist)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for Histogram Difference.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        if (!calculate_median(s->dct_window, s->window_size, &median_dct)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for DCT Energy.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->dct_window, s->window_size, median_dct, &mad_dct)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for DCT Energy.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        if (!calculate_median(s->sobel_window, s->window_size, &median_sobel)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for Sobel Energy.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->sobel_window, s->window_size, median_sobel, &mad_sobel)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for Sobel Energy.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        if (!calculate_median(s->entropy_window, s->window_size, &median_entropy)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for Entropy.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->entropy_window, s->window_size, median_entropy, &mad_entropy)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for Entropy.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        if (!calculate_median(s->color_var_window, s->window_size, &median_color_var)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate median for Color Variance.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }
        if (!calculate_mad(s->color_var_window, s->window_size, median_color_var, &mad_color_var)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate MAD for Color Variance.\n");
            av_frame_free(&gray_frame);
            return AVERROR(ENOMEM);
        }

        END_TIMER(MAD_MEDIAN, "MAD_MEDIAN");

        // Adjust dynamic weights based on delta metrics
        adjust_weights(s, delta_complexity, delta_ssim, delta_hist, delta_dct, delta_sobel, delta_entropy, delta_color_var);

        // Determine dynamic thresholds based on delta metrics
        double threshold_complexity = median_complexity + s->alpha_complexity * fmax(mad_complexity, 1e-6);
        double threshold_ssim = median_ssim + s->alpha_ssim * fmax(mad_ssim, 1e-6);
        double threshold_hist = median_hist + s->alpha_hist * fmax(mad_hist, 1e-6);
        double threshold_dct = median_dct + s->alpha_dct * fmax(mad_dct, 1e-6);
        double threshold_sobel = median_sobel + s->alpha_sobel * fmax(mad_sobel, 1e-6);
        double threshold_entropy = median_entropy + 0.5 * fmax(mad_entropy, 1e-6); // Example alpha for entropy
        double threshold_color_var = median_color_var + 50.0 * fmax(mad_color_var, 1e-6); // Example alpha for color variance

        // Recalculate Adaptive Threshold if weighted_sum_window is filled
        if (s->weighted_sum_window_filled) {
            START_TIMER(ADAPTIVE_THRESHOLD);
            bool threshold_calculated = calculate_adaptive_threshold(s);
            END_TIMER(ADAPTIVE_THRESHOLD, "ADAPTIVE_THRESHOLD");
            if (!threshold_calculated) {
                av_log(ctx, AV_LOG_ERROR, "Failed to calculate adaptive threshold.\n");
                av_frame_free(&gray_frame);
                return AVERROR(ENOMEM);
            }
        }

        // Log detailed information (Removed Duplicate Logging)
        av_log(ctx, AV_LOG_DEBUG, "Frame %lld: Raw Metrics - Complexity: %.2f, SSIM: %.4f, Histogram Difference: %.2f, DCT Energy: %.2f, Sobel Energy: %.2f, Entropy: %.2f, Color Variance: %.2f\n",
               frame->pts, delta_complexity, delta_ssim, delta_hist, delta_dct, delta_sobel, delta_entropy, delta_color_var);

        av_log(ctx, AV_LOG_DEBUG, "Frame %lld: Normalized Metrics - Complexity: %.2f, SSIM: %.2f, Histogram: %.2f, DCT: %.2f, Sobel: %.2f, Entropy: %.2f, Color Var: %.2f\n",
               frame->pts, norm_complexity, norm_ssim, norm_hist, norm_dct, norm_sobel, norm_entropy, norm_color_var);

        av_log(ctx, AV_LOG_DEBUG, "Frame %lld: Weighted_Sum=%.2f, Median_WS=%.2f, MAD_WS=%.2f, Adaptive_Threshold=%.2f\n",
               frame->pts, activity_score, s->median_weighted_sum, s->mad_weighted_sum, s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum);


        // Define thresholds to adjust frame_interval
        // These thresholds can be tuned based on empirical observations
        double high_activity_threshold = s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum;
        double low_activity_threshold = s->median_weighted_sum - s->k_threshold * s->mad_weighted_sum;

        // Adjust frame_interval based on activity_score
        if (s->activity_score > high_activity_threshold && s->frame_interval > s->min_frame_interval) {
            s->frame_interval--;
            av_log(ctx, AV_LOG_DEBUG, "High activity detected. Decreasing frame_interval to %d.\n", s->frame_interval);
        }
        else if (s->activity_score < low_activity_threshold && s->frame_interval < s->max_frame_interval) {
            s->frame_interval++;
            av_log(ctx, AV_LOG_DEBUG, "Low activity detected. Increasing frame_interval to %d.\n", s->frame_interval);
        }
            
        

        // Handle cooldown period
        if (s->current_cooldown > 0) {
            s->current_cooldown--;
            // Reset consecutive detections if in cooldown
            s->consecutive_detected = 0;
        } else {
            bool detected = false;
            if (s->weighted_sum_window_filled) {
                detected = (activity_score > (s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum));
            }

            if (detected) {
                s->consecutive_detected++;

                if (s->consecutive_detected >= s->required_consecutive_changes) {


                    // Calculate dynamic CRF based on activity_score
                    int crf = calculate_dynamic_crf(s, activity_score);
                    av_log(ctx, AV_LOG_INFO, "Dynamic CRF set to %d based on activity score: %.3f\n", crf, activity_score);

                    attach_crf_metadata(frame, crf);

                    // Confirm scene change
                    av_log(ctx, AV_LOG_INFO, "Scene change confirmed: Frame=%lld, Weighted_Sum=%.2f > Adaptive_Threshold=%.2f\n",
                           frame->pts, activity_score, s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum);

                    // Reset consecutive detections and set cooldown
                    s->consecutive_detected = 0;
                    s->current_cooldown = s->cooldown_frames;
                }
            } else {
                // Reset consecutive detections if scene change not detected
                s->consecutive_detected = 0;
            }
        }

        // Update previous complexity
        s->previous_complexity = current_complexity;

        // Store current grayscale frame as previous for next iteration
        // Swap frames instead of copying to optimize performance
        AVFrame *tmp = s->prev_gray_frame;
        s->prev_gray_frame = gray_frame;
        gray_frame = tmp;

        
        END_TIMER(GLOBAL, "GLOBAL");

        // Pass the original frame to the next filter
        return ff_filter_frame(ctx->outputs[0], frame);
    }
    END_TIMER(GLOBAL, "GLOBAL");
    return ff_filter_frame(ctx->outputs[0], frame);
}



/**
 * @brief Calculate dynamic CRF based on activity score.
 *
 * @param s Pointer to CaeContext.
 * @param activity_score Normalized activity score (0 to 1).
 * @return int Calculated CRF value.
 */
static int calculate_dynamic_crf(CaeContext *s, double activity_score) {
    int min_crf = 15;
    int max_crf = 35;

    // Sigmoid parameters
    double k = s->sigmoid_slope;      // Controls the steepness of the curve
    double x0 = s->sigmoid_midpoint;  // Midpoint of the sigmoid curve

    // Sigmoid function scaling
    double scaling_factor = 1.0 / (1.0 + exp(-k * (activity_score - x0)));

    // Invert scaling factor for inverse relationship
    scaling_factor = 1.0 - scaling_factor;

    int crf = min_crf + (int)((max_crf - min_crf) * scaling_factor);
    crf = FFMIN(FFMAX(crf, min_crf), max_crf);

    av_log(s->ctx, AV_LOG_INFO, "Dynamic CRF set to %d based on activity score: %.3f\n", crf, activity_score);

    return crf;
}

/**
 * @brief Attach CRF value to the frame's metadata.
 *
 * @param frame The AVFrame to which the metadata will be attached.
 * @param crf The predicted CRF value to attach.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static int attach_crf_metadata(AVFrame *frame, int crf) {

    if (!frame)
        return 0;


    // Allocate new side data
    AVFrameSideData *side_data = av_frame_new_side_data(frame, AV_FRAME_DATA_CUSTOM_CRF, sizeof(int));
    if (!side_data) {
        // Handle allocation failure
        fprintf(stderr, "Failed to allocate side data for frame.\n");
        return AVERROR(ENOMEM);
    }

    // Copy the CRF value into the side data
    memcpy(side_data->data, &crf, sizeof(int));
        
    return 0;
}



/**
 * @brief Define filter options (configurable parameters).
 */
#define OFFSET(x) offsetof(CaeContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption cae_options[] = {
    { "alpha_complexity", "Threshold multiplier for complexity", OFFSET(alpha_complexity), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "alpha_ssim", "Threshold multiplier for SSIM", OFFSET(alpha_ssim), AV_OPT_TYPE_DOUBLE, {.dbl = 0.8}, 0, 10, FLAGS },
    { "alpha_hist", "Threshold multiplier for Histogram Difference", OFFSET(alpha_hist), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "alpha_dct", "Threshold multiplier for DCT Energy", OFFSET(alpha_dct), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "alpha_sobel", "Threshold multiplier for Sobel Energy", OFFSET(alpha_sobel), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "window_size", "Number of frames in the sliding window", OFFSET(window_size), AV_OPT_TYPE_INT, {.i64 = 30}, 1, 100, FLAGS },
    { "threshold_mode", "Thresholding mode (0: Median + alpha*MAD)", OFFSET(threshold_mode), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 0, FLAGS },
    { "cooldown_frames", "Number of frames to ignore after a scene change", OFFSET(cooldown_frames), AV_OPT_TYPE_INT, {.i64 = 10}, 0, 100, FLAGS },
    { "required_consecutive_changes", "Number of consecutive detections to confirm scene change", OFFSET(required_consecutive_changes), AV_OPT_TYPE_INT, {.i64 = 2}, 1, 10, FLAGS },
    { "k_threshold", "Multiplier for MAD in adaptive threshold calculation", OFFSET(k_threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 3.0}, 1.0, 10.0, FLAGS },
    { "max_weight", "Maximum sum of metric weights to prevent domination", OFFSET(max_weight), AV_OPT_TYPE_DOUBLE, {.dbl = 10.0}, 1.0, 100.0, FLAGS },
    { "min_frame_interval", "Minimum frame interval for adaptive processing", OFFSET(min_frame_interval), AV_OPT_TYPE_INT, {.i64 = 1}, 1, 1000, FLAGS },
    { "max_frame_interval", "Maximum frame interval for adaptive processing", OFFSET(max_frame_interval), AV_OPT_TYPE_INT, {.i64 = 5}, 1, 1000, FLAGS },
    { "weight_complexity", "Weight for complexity metric", OFFSET(weight_complexity), AV_OPT_TYPE_DOUBLE, {.dbl = 1.5}, 0, 10, FLAGS },
    { "weight_ssim", "Weight for SSIM metric", OFFSET(weight_ssim), AV_OPT_TYPE_DOUBLE, {.dbl = 2.0}, 0, 10, FLAGS },
    { "weight_hist", "Weight for histogram difference metric", OFFSET(weight_hist), AV_OPT_TYPE_DOUBLE, {.dbl = 1.0}, 0, 10, FLAGS },
    { "weight_dct", "Weight for DCT energy metric", OFFSET(weight_dct), AV_OPT_TYPE_DOUBLE, {.dbl = 1.5}, 0, 10, FLAGS },
    { "weight_sobel", "Weight for Sobel energy metric", OFFSET(weight_sobel), AV_OPT_TYPE_DOUBLE, {.dbl = 1.0}, 0, 10, FLAGS },
    { "weight_entropy", "Weight for entropy metric", OFFSET(weight_entropy), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "weight_color_var", "Weight for color variance metric", OFFSET(weight_color_var), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "crf_exponent", "Exponent for CRF scaling", OFFSET(crf_exponent), AV_OPT_TYPE_DOUBLE, {.dbl = 2.0}, 0.1, 5.0, FLAGS },
    { "sigmoid_slope", "Slope of the sigmoid function", OFFSET(sigmoid_slope), AV_OPT_TYPE_DOUBLE, {.dbl = 10.0}, 0.1, 100.0, FLAGS },
    { "sigmoid_midpoint", "Midpoint of the sigmoid function", OFFSET(sigmoid_midpoint), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0.0, 1.0, FLAGS },
    { NULL }
};

/**
 * @brief Define supported pixel formats.
 */
static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_GRAY8, // Added grayscale
    AV_PIX_FMT_NONE
};

/**
 * @brief Define the AVClass for the CAE filter.
 */
AVFILTER_DEFINE_CLASS(cae);

/**
 * @brief Define the input pads for the CAE filter.
 */
static const AVFilterPad avfilter_vf_cae_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = cae_config_props,
    }
};

/**
 * @brief Register the CAE filter with FFmpeg.
 */
const AVFilter ff_vf_cae = {
    .name          = "cae",
    .description   = NULL_IF_CONFIG_SMALL("Detect scene changes based on frame metrics (DCT, Sobel, SAD, SSIM, Histogram Difference) with dynamic weighting and dynamic adaptive threshold."),
    .priv_size     = sizeof(CaeContext),
    .priv_class    = &cae_class,
    .init          = init_filter,
    .uninit        = uninit_filter,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    FILTER_INPUTS(avfilter_vf_cae_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
