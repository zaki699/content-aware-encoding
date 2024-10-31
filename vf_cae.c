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

#include <fftw3.h>    // For DCT computations

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define BLOCK_SIZE 16 // BLOCK SIZE


typedef struct CaeContext {
    const AVClass *class;

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
static double compute_Sobel_energy_NEON(const uint8_t *data, int width, int height, int stride);
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
    int private_hists[num_threads][256];
    memset(private_hists, 0, sizeof(private_hists));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int *local_hist = private_hists[thread_id];

        #pragma omp for
        for (int y = 0; y < height; y++) {
            const uint8_t *row = data + y * stride;

            // Prefetch the row
            __builtin_prefetch(row, 0, 3);

            int x = 0;
            for (; x <= width - 16; x += 16) {
                uint8x16_t pixels = vld1q_u8(row + x);

                uint8_t pixels_array[16];
                vst1q_u8(pixels_array, pixels);
                for (int i = 0; i < 16; i++) {
                    local_hist[pixels_array[i]]++;
                }
            }

            for (; x < width; x++) {
                local_hist[row[x]]++;
            }
        }
    }

    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < 256; j++) {
            hist[j] += private_hists[i][j];
        }
    }
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
 * @brief Function to compute SSIM using NEON intrinsics (Partial Optimization)
 */
static double compute_SSIM_NEON(const uint8_t *prev, const uint8_t *curr, int width, int height, int stride) {
#ifdef __ARM_NEON
    const float C1 = 6.5025f, C2 = 58.5225f;
    const float pixel_count_inv = 1.0f / (width * height);

    float mean_prev = 0.0f, mean_curr = 0.0f;
    float variance_prev = 0.0f, variance_curr = 0.0f, covariance = 0.0f;

    // First pass: Calculate means
    for (int y = 0; y < height; y++) {
        const uint8_t *prev_row = prev + y * stride;
        const uint8_t *curr_row = curr + y * stride;

        // Prefetch rows to improve cache performance
        __builtin_prefetch(prev_row, 0, 3);
        __builtin_prefetch(curr_row, 0, 3);

        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16_t p_vals = vld1q_u8(prev_row + x);
            uint8x16_t c_vals = vld1q_u8(curr_row + x);

            uint16x8_t p_low = vmovl_u8(vget_low_u8(p_vals));
            uint16x8_t p_high = vmovl_u8(vget_high_u8(p_vals));
            uint16x8_t c_low = vmovl_u8(vget_low_u8(c_vals));
            uint16x8_t c_high = vmovl_u8(vget_high_u8(c_vals));

            mean_prev += vaddvq_u32(vpaddlq_u16(p_low)) + vaddvq_u32(vpaddlq_u16(p_high));
            mean_curr += vaddvq_u32(vpaddlq_u16(c_low)) + vaddvq_u32(vpaddlq_u16(c_high));
        }

        for (; x < width; x++) {
            mean_prev += prev_row[x];
            mean_curr += curr_row[x];
        }
    }

    mean_prev *= pixel_count_inv;
    mean_curr *= pixel_count_inv;

    // Second pass: Calculate variances and covariance
    for (int y = 0; y < height; y++) {
        const uint8_t *prev_row = prev + y * stride;
        const uint8_t *curr_row = curr + y * stride;

        // Prefetch rows to improve cache performance
        __builtin_prefetch(prev_row, 0, 3);
        __builtin_prefetch(curr_row, 0, 3);

        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16_t p_vals = vld1q_u8(prev_row + x);
            uint8x16_t c_vals = vld1q_u8(curr_row + x);

            float32x4_t dp_low = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u8(p_vals))), vdupq_n_f32(mean_prev));
            float32x4_t dp_high = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u8(p_vals))), vdupq_n_f32(mean_prev));
            float32x4_t dc_low = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u8(c_vals))), vdupq_n_f32(mean_curr));
            float32x4_t dc_high = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u8(c_vals))), vdupq_n_f32(mean_curr));

            variance_prev += vaddvq_f32(vmulq_f32(dp_low, dp_low)) + vaddvq_f32(vmulq_f32(dp_high, dp_high));
            variance_curr += vaddvq_f32(vmulq_f32(dc_low, dc_low)) + vaddvq_f32(vmulq_f32(dc_high, dc_high));
            covariance += vaddvq_f32(vmulq_f32(dp_low, dc_low)) + vaddvq_f32(vmulq_f32(dp_high, dc_high));
        }

        for (; x < width; x++) {
            float dp = (float)prev_row[x] - mean_prev;
            float dc = (float)curr_row[x] - mean_curr;
            variance_prev += dp * dp;
            variance_curr += dc * dc;
            covariance += dp * dc;
        }
    }

    variance_prev *= pixel_count_inv;
    variance_curr *= pixel_count_inv;
    covariance *= pixel_count_inv;

    float numerator = (2 * mean_prev * mean_curr + C1) * (2 * covariance + C2);
    float denominator = (mean_prev * mean_prev + mean_curr * mean_curr + C1) * (variance_prev + variance_curr + C2);

    return (double)(numerator / denominator);
#else
    return compute_SSIM(prev, curr, width, height, stride);
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
 * @brief Compute Sobel Energy in blocks using NEON
 * 
 * @param data Pointer to the input image data
 * @param width Width of the image
 * @param height Height of the image
 * @param stride Stride of the image (number of bytes per row)
 * @return double The total Sobel energy for the entire image
 */
static double compute_Sobel_energy_NEON(const uint8_t *data, int width, int height, int stride) {
#ifdef __ARM_NEON
    double total_sobel_energy = 0.0;

    // Define Sobel kernels for x and y directions
    int8x8_t sobel_x = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int8x8_t sobel_y = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    // Process image in BLOCK_SIZE x BLOCK_SIZE blocks
    for (int y = 1; y < height - 1; y += BLOCK_SIZE) {
        for (int x = 1; x < width - 1; x += BLOCK_SIZE) {
            float block_sobel_energy = 0.0f;

            for (int by = 0; by < BLOCK_SIZE && (y + by) < height - 1; by++) {
                const uint8_t *row = data + (y + by) * stride + x;

                // Prefetch next row to cache
                __builtin_prefetch(row + stride, 0, 1);

                for (int bx = 0; bx < BLOCK_SIZE && (x + bx) < width - 1; bx += 8) {
                    // Load three rows around the target pixel for Sobel calculation
                    const uint8_t *top = row + bx - stride;
                    const uint8_t *mid = row + bx;
                    const uint8_t *bottom = row + bx + stride;

                    uint8x8_t t = vld1_u8(top);
                    uint8x8_t m = vld1_u8(mid);
                    uint8x8_t b = vld1_u8(bottom);

                    // Calculate gradients in x and y directions
                    int16x8_t gx = vmlal_s8(vmlal_s8(vdupq_n_s16(0), vreinterpret_s8_u8(vsub_u8(vext_u8(t, t, 1), vext_u8(t, t, 2))), sobel_x),
                                            vreinterpret_s8_u8(vsub_u8(vext_u8(m, m, 1), vext_u8(m, m, 2))), sobel_x);
                    gx = vmlal_s8(gx, vreinterpret_s8_u8(vsub_u8(vext_u8(b, b, 1), vext_u8(b, b, 2))), sobel_x);

                    int16x8_t gy = vmlal_s8(vmlal_s8(vdupq_n_s16(0), vreinterpret_s8_u8(vsub_u8(vext_u8(t, b, 1), vext_u8(b, b, 2))), sobel_y),
                                            vreinterpret_s8_u8(vsub_u8(vext_u8(t, t, 1), vext_u8(t, t, 2))), sobel_y);
                    gy = vmlal_s8(gy, vreinterpret_s8_u8(vsub_u8(vext_u8(b, b, 1), vext_u8(b, b, 2))), sobel_y);

                    // Calculate magnitude of gradients
                    int32x4_t mag_low = vaddl_s16(vget_low_s16(gx), vget_low_s16(gy));
                    int32x4_t mag_high = vaddl_s16(vget_high_s16(gx), vget_high_s16(gy));

                    // Sum up the results to calculate block energy
                    block_sobel_energy += vaddvq_f32(vcvtq_f32_s32(mag_low)) + vaddvq_f32(vcvtq_f32_s32(mag_high));

                    // Prefetch next set of data for the bottom row
                    __builtin_prefetch(bottom + bx + stride, 0, 1);
                }
            }
            total_sobel_energy += block_sobel_energy;
        }
    }
    return total_sobel_energy;
#else
    // Fallback to scalar Sobel if NEON is not available
    return compute_sobel_energy(data, width, height, stride);
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
 * @brief Compute entropy in BLOCK_SIZE x BLOCK_SIZE blocks using NEON
 *
 * @param data Pointer to the input image data
 * @param width Width of the image
 * @param height Height of the image
 * @param stride Stride of the image (number of bytes per row)
 * @return double The total entropy for the entire image
 */
static double compute_entropy_NEON(const uint8_t *data, int width, int height, int stride) {
#ifdef __ARM_NEON
    double total_entropy = 0.0;

    // Iterate over the image in BLOCK_SIZE x BLOCK_SIZE blocks
    for (int y = 0; y < height; y += BLOCK_SIZE) {
        for (int x = 0; x < width; x += BLOCK_SIZE) {
            int hist[256] = {0}; // Initialize histogram for the block

            // Process each row within the block
            for (int by = 0; by < BLOCK_SIZE && (y + by) < height; by++) {
                const uint8_t *row = data + (y + by) * stride + x;
                int bx = 0;

                // Prefetch the row data to improve cache performance
                __builtin_prefetch(row, 0, 3);

                // Process 16 pixels at a time using NEON
                for (; bx <= BLOCK_SIZE - 16 && (x + bx) < width; bx += 16) {
                    uint8x16_t pixels = vld1q_u8(row + bx);

                    // Increment histogram values for each pixel by manually unrolling
                    hist[vgetq_lane_u8(pixels, 0)]++;
                    hist[vgetq_lane_u8(pixels, 1)]++;
                    hist[vgetq_lane_u8(pixels, 2)]++;
                    hist[vgetq_lane_u8(pixels, 3)]++;
                    hist[vgetq_lane_u8(pixels, 4)]++;
                    hist[vgetq_lane_u8(pixels, 5)]++;
                    hist[vgetq_lane_u8(pixels, 6)]++;
                    hist[vgetq_lane_u8(pixels, 7)]++;
                    hist[vgetq_lane_u8(pixels, 8)]++;
                    hist[vgetq_lane_u8(pixels, 9)]++;
                    hist[vgetq_lane_u8(pixels, 10)]++;
                    hist[vgetq_lane_u8(pixels, 11)]++;
                    hist[vgetq_lane_u8(pixels, 12)]++;
                    hist[vgetq_lane_u8(pixels, 13)]++;
                    hist[vgetq_lane_u8(pixels, 14)]++;
                    hist[vgetq_lane_u8(pixels, 15)]++;
                }

                // Process remaining pixels in the row
                for (; bx < BLOCK_SIZE && (x + bx) < width; bx++) {
                    hist[row[bx]]++;
                }
            }

            // Calculate entropy for the current block
            double block_entropy = 0.0;
            int pixel_count = BLOCK_SIZE * BLOCK_SIZE;

            // Avoid division by zero in case block area is smaller
            if (pixel_count > 0) {
                for (int i = 0; i < 256; i++) {
                    if (hist[i] > 0) {
                        double probability = (double)hist[i] / pixel_count;
                        block_entropy -= probability * log2(probability);
                    }
                }
            }

            total_entropy += block_entropy;
        }
    }

    return total_entropy;
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

    for (int y = 0; y < height; y += BLOCK_SIZE) {
        for (int x = 0; x < width; x += BLOCK_SIZE) {

            // Iterate through each row in the block
            for (int by = 0; by < BLOCK_SIZE && (y + by) < height; by++) {
                const uint8_t *row = data + (y + by) * stride + x;

                // Prefetch row data to improve cache performance
                __builtin_prefetch(row, 0, 3);

                int bx = 0;

                float32x4_t v_sum = vdupq_n_f32(0.0f);
                float32x4_t v_sum_sq = vdupq_n_f32(0.0f);

                // Process 16 pixels at a time
                for (; bx <= BLOCK_SIZE - 16 && (x + bx) < width; bx += 16) {
                    uint8x16_t pixels = vld1q_u8(row + bx);

                    uint16x8_t pixels_low = vmovl_u8(vget_low_u8(pixels));
                    uint16x8_t pixels_high = vmovl_u8(vget_high_u8(pixels));

                    float32x4_t p1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixels_low)));
                    float32x4_t p2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(pixels_low)));
                    float32x4_t p3 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixels_high)));
                    float32x4_t p4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(pixels_high)));

                    v_sum = vaddq_f32(v_sum, vaddq_f32(vaddq_f32(p1, p2), vaddq_f32(p3, p4)));
                    v_sum_sq = vaddq_f32(v_sum_sq, vaddq_f32(vaddq_f32(vmulq_f32(p1, p1), vmulq_f32(p2, p2)), 
                                                           vaddq_f32(vmulq_f32(p3, p3), vmulq_f32(p4, p4))));
                }

                sum += vaddvq_f32(v_sum);
                sum_sq += vaddvq_f32(v_sum_sq);

                // Handle any remaining pixels
                for (; bx < BLOCK_SIZE && (x + bx) < width; bx++) {
                    float pixel = (float)row[bx];
                    sum += pixel;
                    sum_sq += pixel * pixel;
                }
            }
        }
    }

    double mean = sum / (width * height);
    double variance = (sum_sq / (width * height)) - (mean * mean);
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
    s->prev_gray_frame = av_frame_alloc();
    if (!s->prev_gray_frame) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate AVFrame for previous grayscale data.\n");
        sws_freeContext(s->sws_ctx);
        return AVERROR(ENOMEM);
    }

    // Configure the previous grayscale frame
    s->prev_gray_frame->format = s->dst_pix_fmt;
    s->prev_gray_frame->width  = s->width;
    s->prev_gray_frame->height = s->height;

    if (av_frame_get_buffer(s->prev_gray_frame, 32) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate buffer for previous grayscale frame.\n");
        av_frame_free(&s->prev_gray_frame);
        sws_freeContext(s->sws_ctx);
        return AVERROR(ENOMEM);
    }

    av_frame_make_writable(s->prev_gray_frame);
    memset(s->prev_gray_frame->data[0], 0, s->prev_gray_frame->linesize[0] * s->height);

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
        s->window_size = 30;
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
    s->weight_complexity = 1.0;
    s->weight_ssim = 0.8;
    s->weight_hist = 0.7;
    s->weight_dct = 1.2;
    s->weight_sobel = 1.0;
    s->weight_entropy = 0.5;
    s->weight_color_var = 0.5;

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
    s->median_weighted_sum = 0.0;
    s->mad_weighted_sum = 0.0;

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
 * @param inlink The input link.
 * @param frame The input frame.
 * @return int 0 on success, negative AVERROR on failure.
 */
static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
    AVFilterContext *ctx = inlink->dst;
    CaeContext *s = ctx->priv;

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
        current_complexity = compute_SAD_NEON(
            s->prev_gray_frame->data[0],   // Previous grayscale frame data
            gray_frame->data[0],           // Current grayscale frame data
            s->width,
            s->height,
            gray_frame->linesize[0]
        );
    }

    // Compute Histograms
    compute_histogram_NEON(s->prev_gray_frame->data[0], s->width, s->height, gray_frame->linesize[0], s->hist_prev);
    compute_histogram_NEON(gray_frame->data[0], s->width, s->height, gray_frame->linesize[0], s->hist_curr);

    // Compute Histogram Difference
    double hist_diff = compute_hist_diff(s->hist_prev, s->hist_curr);

    // Compute SSIM
    double ssim = compute_SSIM_NEON(
        s->prev_gray_frame->data[0],
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );


    // Compute DCT Energy
    double dct_energy = compute_DCT_energy(
        s,
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );

    // Compute Sobel Energy (Can also be optimized similarly if desired)
    double sobel_energy = compute_Sobel_energy_NEON(
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );

    // Compute Entropy
    double entropy = compute_entropy_NEON(
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );

    // Compute Color Variance
    double color_variance = compute_color_variance_NEON(
        gray_frame->data[0],
        s->width,
        s->height,
        gray_frame->linesize[0]
    );


    // Compute delta metrics
    double delta_complexity = fabs(current_complexity - s->previous_complexity);
    double delta_ssim = fabs(ssim - 1.0); // SSIM ranges from 0 to 1
    double delta_hist = hist_diff;
    double delta_dct = dct_energy;       // Assuming higher DCT energy indicates more change
    double delta_sobel = sobel_energy;   // Assuming higher Sobel energy indicates more change
    double delta_entropy = entropy;      // Assuming higher entropy indicates more change
    double delta_color_var = color_variance; // Assuming higher color variance indicates more change

    // Normalize metrics using logarithmic scaling
    double norm_complexity = log(delta_complexity + 1.0);
    double norm_ssim = log(delta_ssim + 1.0);
    double norm_hist = log(delta_hist + 1.0);
    double norm_dct = log(delta_dct + 1.0);
    double norm_sobel = log(delta_sobel + 1.0);
    double norm_entropy = log(delta_entropy + 1.0);
    double norm_color_var = log(delta_color_var + 1.0);

    // Compute weighted sum with normalized metrics
    double weighted_sum = (norm_complexity * s->weight_complexity) +
                          (norm_ssim * s->weight_ssim) +
                          (norm_hist * s->weight_hist) +
                          (norm_dct * s->weight_dct) +
                          (norm_sobel * s->weight_sobel) +
                          (norm_entropy * s->weight_entropy) +
                          (norm_color_var * s->weight_color_var);

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
    s->weighted_sum_window[s->weighted_sum_window_index] = weighted_sum;
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
            bool threshold_calculated = calculate_adaptive_threshold(s);
            if (!threshold_calculated) {
                av_log(ctx, AV_LOG_ERROR, "Failed to calculate adaptive threshold.\n");
                av_frame_free(&gray_frame);
                return AVERROR(ENOMEM);
            }
        }

        // Log detailed information
        av_log(ctx, AV_LOG_DEBUG, "Frame %lld: Delta_C=%.2f, Delta_S=%.4f, Delta_H=%.2f, Delta_DCT=%.2f, Delta_Sobel=%.2f, Delta_E=%.2f, Delta_CV=%.2f, "
               "Median_C=%.2f, MAD_C=%.2f, Threshold_C=%.2f, "
               "Median_S=%.4f, MAD_S=%.4f, Threshold_S=%.2f, "
               "Median_H=%.2f, MAD_H=%.2f, Threshold_H=%.2f, "
               "Median_DCT=%.2f, MAD_DCT=%.2f, Threshold_DCT=%.2f, "
               "Median_Sobel=%.2f, MAD_Sobel=%.2f, Threshold_Sobel=%.2f, "
               "Median_E=%.2f, MAD_E=%.2f, Threshold_E=%.2f, "
               "Median_CV=%.2f, MAD_CV=%.2f, Threshold_CV=%.2f, "
               "Weighted_Sum=%.2f, Median_WS=%.2f, MAD_WS=%.2f, Adaptive_Threshold=%.2f\n",
               frame->pts, delta_complexity, delta_ssim, delta_hist, delta_dct, delta_sobel, delta_entropy, delta_color_var,
               median_complexity, mad_complexity, threshold_complexity,
               median_ssim, mad_ssim, threshold_ssim,
               median_hist, mad_hist, threshold_hist,
               median_dct, mad_dct, threshold_dct,
               median_sobel, mad_sobel, threshold_sobel,
               median_entropy, mad_entropy, threshold_entropy,
               median_color_var, mad_color_var, threshold_color_var,
               weighted_sum, s->median_weighted_sum, s->mad_weighted_sum, s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum);

        // Handle cooldown period
        if (s->current_cooldown > 0) {
            s->current_cooldown--;
            // Reset consecutive detections if in cooldown
            s->consecutive_detected = 0;
        } else {
            bool detected = false;
            if (s->weighted_sum_window_filled) {
                detected = (weighted_sum > (s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum));
            }

            if (detected) {
                s->consecutive_detected++;

                if (s->consecutive_detected >= s->required_consecutive_changes) {
                    // Confirm scene change
                    av_log(ctx, AV_LOG_INFO, "Scene change confirmed: Frame=%lld, Weighted_Sum=%.2f > Adaptive_Threshold=%.2f\n",
                           frame->pts, weighted_sum, s->median_weighted_sum + s->k_threshold * s->mad_weighted_sum);

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

        // Pass the original frame to the next filter
        return ff_filter_frame(ctx->outputs[0], frame);
    }
    return ff_filter_frame(ctx->outputs[0], frame);
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
