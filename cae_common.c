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

#include "cae.h"
#include <time.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif


/* Function Implementations */

// Function to calculate motion vector magnitude
static int calculate_motion_vector_magnitude(const AVFrame *frame, double *motion_magnitude) {
    if (!frame || !motion_magnitude) {
        av_log(NULL, AV_LOG_ERROR, "calculate_motion_vector_magnitude: Invalid input\n");
        return AVERROR(EINVAL);
    }

    const AVFrameSideData *side_data = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (!side_data) {
        *motion_magnitude = 0.0;
        return 0; // No motion vectors present, consider zero magnitude
    }

    const AVMotionVector *mvs = (const AVMotionVector *)side_data->data;
    int mv_count = side_data->size / sizeof(AVMotionVector);

    if (mv_count == 0) {
        *motion_magnitude = 0.0;
        return 0;
    }

    double total_magnitude = 0.0;

    #pragma omp parallel for reduction(+:total_magnitude)
    for (int i = 0; i < mv_count; i++) {
        double motion_x = mvs[i].motion_x / 4.0; // motion vectors are in quarter-pixel units
        double motion_y = mvs[i].motion_y / 4.0;
        double magnitude = sqrt(motion_x * motion_x + motion_y * motion_y);
        total_magnitude += magnitude;
    }

    *motion_magnitude = total_magnitude / mv_count;
    av_log(NULL, AV_LOG_DEBUG, "Motion vector magnitude: %f\n", *motion_magnitude);
    return 0;
}

// Function to calculate edge density using Sobel filter
static int calculate_edge_density(const AVFrame *frame, double *edge_density) {
    if (!frame || !edge_density) {
        av_log(NULL, AV_LOG_ERROR, "calculate_edge_density: Invalid input\n");
        return AVERROR(EINVAL);
    }

    int width = frame->width;
    int height = frame->height;

    if (frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_edge_density: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *y_plane = frame->data[0];
    int stride = frame->linesize[0];
    int edge_count = 0;
    int total_pixels = (width - 2) * (height - 2);

    #pragma omp parallel for reduction(+:edge_count)
    for (int y = 1; y < height - 1; y++) {
        uint8_t *row_prev = &y_plane[(y - 1) * stride];
        uint8_t *row_curr = &y_plane[y * stride];
        uint8_t *row_next = &y_plane[(y + 1) * stride];

        for (int x = 1; x < width - 1; x++) {
            int gx = -row_prev[x - 1] - 2 * row_curr[x - 1] - row_next[x - 1]
                     + row_prev[x + 1] + 2 * row_curr[x + 1] + row_next[x + 1];
            int gy = -row_prev[x - 1] - 2 * row_prev[x] - row_prev[x + 1]
                     + row_next[x - 1] + 2 * row_next[x] + row_next[x + 1];
            int magnitude = abs(gx) + abs(gy);
            if (magnitude > EDGE_DETECTION_THRESHOLD) {
                edge_count++;
            }
        }
    }

    *edge_density = (double)edge_count / total_pixels;
    av_log(NULL, AV_LOG_DEBUG, "Edge density: %f\n", *edge_density);
    return 0;
}

// Function to calculate histogram complexity
static int calculate_histogram_complexity(const AVFrame *frame, double *histogram_complexity) {
    if (!frame || !histogram_complexity) {
        av_log(NULL, AV_LOG_ERROR, "calculate_histogram_complexity: Invalid input\n");
        return AVERROR(EINVAL);
    }

    int width = frame->width;
    int height = frame->height;

    if (frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_histogram_complexity: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *y_plane = frame->data[0];
    int stride = frame->linesize[0];
    int histogram[256] = {0};
    int num_pixels = width * height;

    #pragma omp parallel
    {
        int local_histogram[256] = {0};
        #pragma omp for nowait
        for (int y = 0; y < height; y++) {
            uint8_t *row = &y_plane[y * stride];
            for (int x = 0; x < width; x++) {
                int pixel_value = row[x];
                local_histogram[pixel_value]++;
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++) {
                histogram[i] += local_histogram[i];
            }
        }
    }

    double mean = 0.0;
    for (int i = 0; i < 256; i++) {
        mean += i * histogram[i];
    }
    mean /= num_pixels;

    double variance = 0.0;
    for (int i = 0; i < 256; i++) {
        variance += histogram[i] * pow(i - mean, 2);
    }
    variance /= num_pixels;

    *histogram_complexity = variance;
    av_log(NULL, AV_LOG_DEBUG, "Histogram complexity: %f\n", *histogram_complexity);
    return 0;
}

// Function to calculate temporal difference between frames
static int calculate_temporal_difference(const AVFrame *current_frame, const AVFrame *previous_frame, double *temporal_difference) {
    if (!current_frame || !previous_frame || !temporal_difference) {
        av_log(NULL, AV_LOG_ERROR, "calculate_temporal_difference: Invalid input\n");
        return AVERROR(EINVAL);
    }

    if (current_frame->width != previous_frame->width ||
        current_frame->height != previous_frame->height ||
        current_frame->format != previous_frame->format) {
        av_log(NULL, AV_LOG_ERROR, "calculate_temporal_difference: Frame format or size mismatch\n");
        return AVERROR(EINVAL);
    }

    int width = current_frame->width;
    int height = current_frame->height;

    if (current_frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_temporal_difference: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *curr_y_plane = current_frame->data[0];
    uint8_t *prev_y_plane = previous_frame->data[0];
    int stride = current_frame->linesize[0];
    double total_diff = 0.0;

    #pragma omp parallel for reduction(+:total_diff)
    for (int y = 0; y < height; y++) {
        uint8_t *curr_row = &curr_y_plane[y * stride];
        uint8_t *prev_row = &prev_y_plane[y * stride];
        for (int x = 0; x < width; x++) {
            total_diff += abs(curr_row[x] - prev_row[x]);
        }
    }

    *temporal_difference = total_diff / (width * height);
    av_log(NULL, AV_LOG_DEBUG, "Temporal difference: %f\n", *temporal_difference);
    return 0;
}

/**
 * @brief Calculate Color Variance of the frame.
 *
 * @param frame Input frame.
 * @param color_variance Pointer to store the calculated color variance.
 * @return int 0 on success, negative AVERROR code on failure.
 */
// Function to calculate color variance
static int calculate_color_variance(const AVFrame *frame, double *color_variance) {
    if (!frame || !color_variance) {
        av_log(NULL, AV_LOG_ERROR, "calculate_color_variance: Invalid input\n");
        return AVERROR(EINVAL);
    }

    int width = frame->width;
    int height = frame->height;

    if (frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_color_variance: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *u_plane = frame->data[1];
    uint8_t *v_plane = frame->data[2];
    int stride_u = frame->linesize[1];
    int stride_v = frame->linesize[2];
    double sum_u = 0.0, sum_v = 0.0;
    double sum_sq_u = 0.0, sum_sq_v = 0.0;
    int num_pixels = (width / 2) * (height / 2); // For YUV420p

    #pragma omp parallel for reduction(+:sum_u,sum_v,sum_sq_u,sum_sq_v)
    for (int y = 0; y < height / 2; y++) {
        uint8_t *u_row = &u_plane[y * stride_u];
        uint8_t *v_row = &v_plane[y * stride_v];
        for (int x = 0; x < width / 2; x++) {
            double u = (double)u_row[x];
            double v = (double)v_row[x];
            sum_u += u;
            sum_v += v;
            sum_sq_u += u * u;
            sum_sq_v += v * v;
        }
    }

    double mean_u = sum_u / num_pixels;
    double mean_v = sum_v / num_pixels;
    double variance_u = (sum_sq_u / num_pixels) - (mean_u * mean_u);
    double variance_v = (sum_sq_v / num_pixels) - (mean_v * mean_v);

    *color_variance = (variance_u + variance_v) / 2.0;
    av_log(NULL, AV_LOG_DEBUG, "Color Variance: %f\n", *color_variance);
    return 0;
}

/**
 * @brief Calculate Blockiness of the frame.
 *
 * @param frame Input frame.
 * @param blockiness Pointer to store the calculated blockiness.
 * @return int 0 on success, negative AVERROR code on failure.
 */
// Function to calculate blockiness
static int calculate_blockiness(const AVFrame *frame, double *blockiness) {
    if (!frame || !blockiness) {
        av_log(NULL, AV_LOG_ERROR, "calculate_blockiness: Invalid input\n");
        return AVERROR(EINVAL);
    }

    int width = frame->width;
    int height = frame->height;

    if (frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_blockiness: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *y_plane = frame->data[0];
    int stride = frame->linesize[0];
    double total_blockiness = 0.0;
    int block_count = 0;

    #pragma omp parallel for reduction(+:total_blockiness,block_count)
    for (int y = 0; y < height - 1; y++) {
        uint8_t *row = &y_plane[y * stride];
        uint8_t *row_next = &y_plane[(y + 1) * stride];
        for (int x = 0; x < width - 1; x++) {
            int diff = abs(row[x] - row[x + 1]) + abs(row[x] - row_next[x]);
            total_blockiness += diff;
            block_count += 2;
        }
    }

    *blockiness = total_blockiness / block_count;
    av_log(NULL, AV_LOG_DEBUG, "Blockiness: %f\n", *blockiness);
    return 0;
}

/**
 * @brief Calculate Entropy of the frame.
 *
 * @param frame Input frame.
 * @param entropy Pointer to store the calculated entropy.
 * @return int 0 on success, negative AVERROR code on failure.
 */
// Function to calculate entropy
static int calculate_entropy(const AVFrame *frame, double *entropy) {
    if (!frame || !entropy) {
        av_log(NULL, AV_LOG_ERROR, "calculate_entropy: Invalid input\n");
        return AVERROR(EINVAL);
    }

    int width = frame->width;
    int height = frame->height;

    if (frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_entropy: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *y_plane = frame->data[0];
    int stride = frame->linesize[0];
    int histogram[256] = {0};
    int num_pixels = width * height;

    #pragma omp parallel
    {
        int local_histogram[256] = {0};
        #pragma omp for nowait
        for (int y = 0; y < height; y++) {
            uint8_t *row = &y_plane[y * stride];
            for (int x = 0; x < width; x++) {
                int pixel = row[x];
                local_histogram[pixel]++;
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++) {
                histogram[i] += local_histogram[i];
            }
        }
    }

    double entropy_val = 0.0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            double p = (double)histogram[i] / num_pixels;
            entropy_val -= p * log2(p);
        }
    }

    *entropy = entropy_val;
    av_log(NULL, AV_LOG_DEBUG, "Entropy: %f\n", *entropy);
    return 0;
}

// Function to calculate texture complexity using GLCM
static int calculate_texture_complexity(const AVFrame *frame, double *texture_complexity) {
    if (!frame || !texture_complexity) {
        av_log(NULL, AV_LOG_ERROR, "calculate_texture_complexity: Invalid input\n");
        return AVERROR(EINVAL);
    }

    int width = frame->width;
    int height = frame->height;

    if (frame->format != AV_PIX_FMT_YUV420P) {
        av_log(NULL, AV_LOG_ERROR, "calculate_texture_complexity: Unsupported pixel format\n");
        return AVERROR(EINVAL);
    }

    uint8_t *y_plane = frame->data[0];
    int stride = frame->linesize[0];

    // Allocate GLCM matrix dynamically
    int **glcm = av_malloc_array(GLCM_MATRIX_SIZE, sizeof(int *));
    if (!glcm) {
        av_log(NULL, AV_LOG_ERROR, "calculate_texture_complexity: Memory allocation failed\n");
        return AVERROR(ENOMEM);
    }
    for (int i = 0; i < GLCM_MATRIX_SIZE; i++) {
        glcm[i] = av_calloc(GLCM_MATRIX_SIZE, sizeof(int));
        if (!glcm[i]) {
            av_log(NULL, AV_LOG_ERROR, "calculate_texture_complexity: Memory allocation failed\n");
            for (int j = 0; j < i; j++) {
                av_free(glcm[j]);
            }
            av_free(glcm);
            return AVERROR(ENOMEM);
        }
    }

    // Calculate GLCM for neighboring pixels
    #pragma omp parallel
    {
        int **local_glcm = av_malloc_array(GLCM_MATRIX_SIZE, sizeof(int *));
        for (int i = 0; i < GLCM_MATRIX_SIZE; i++) {
            local_glcm[i] = av_calloc(GLCM_MATRIX_SIZE, sizeof(int));
        }

        #pragma omp for nowait
        for (int y = 0; y < height - 1; y++) {
            uint8_t *row = &y_plane[y * stride];
            uint8_t *row_next = &y_plane[(y + 1) * stride];
            for (int x = 0; x < width - 1; x++) {
                int current_pixel = row[x];
                int neighbor_pixel = row[x + 1];
                local_glcm[current_pixel][neighbor_pixel]++;
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < GLCM_MATRIX_SIZE; i++) {
                for (int j = 0; j < GLCM_MATRIX_SIZE; j++) {
                    glcm[i][j] += local_glcm[i][j];
                }
            }
        }

        for (int i = 0; i < GLCM_MATRIX_SIZE; i++) {
            av_free(local_glcm[i]);
        }
        av_free(local_glcm);
    }

    // Calculate energy and contrast from GLCM
    double energy = 0.0;
    double contrast = 0.0;
    for (int i = 0; i < GLCM_MATRIX_SIZE; i++) {
        for (int j = 0; j < GLCM_MATRIX_SIZE; j++) {
            double value = glcm[i][j];
            energy += value * value;
            contrast += (i - j) * (i - j) * value;
        }
    }

    *texture_complexity = contrast / (energy + 1e-6); // Adding small value to avoid division by zero
    av_log(NULL, AV_LOG_DEBUG, "Texture complexity: %f\n", *texture_complexity);

    // Free GLCM matrix
    for (int i = 0; i < GLCM_MATRIX_SIZE; i++) {
        av_free(glcm[i]);
    }
    av_free(glcm);

    return 0;
}

/**
 * @brief Calculate and log Entropy.
 */
int calculate_and_log_entropy(AVFilterContext *ctx, const AVFrame *frame, double *entropy) {
    if (!entropy) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Entropy.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Entropy calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_entropy(frame,entropy);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Entropy calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Entropy", *entropy) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Entropy = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *entropy, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Blockiness.
 */
int calculate_and_log_blockiness(AVFilterContext *ctx, const AVFrame *frame, double *blockiness) {
    if (!blockiness) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Blockiness.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Blockiness calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_blockiness(frame,blockiness);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Blockiness calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Blockiness", *blockiness) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Blockiness = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *blockiness, time_taken);
    return 0;
}

/**
 * @brief Calculate and log DCT Energy.
 */
// Calculate and log functions (similar adjustments made for all)
int calculate_and_log_dct_energy(AVFilterContext *ctx, const AVFrame *frame, int dct_size, double *dct_energy) {
    if (!dct_energy) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for DCT energy.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for DCT energy calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_frame_dct_energy(frame, dct_size, dct_energy);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "DCT energy calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "DCT energy", *dct_energy) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: DCT Energy = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *dct_energy, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Sobel Energy.
 */
int calculate_and_log_sobel_energy(AVFilterContext *ctx, const AVFrame *frame, int block_size, float *sobel_energy) {
    if (!sobel_energy) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Sobel energy.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Sobel energy calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_frame_sobel_energy(frame, block_size, sobel_energy);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Sobel energy calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Sobel energy", *sobel_energy) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Sobel Energy = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *sobel_energy, time_taken);
    return 0;
}

/**
 * @brief Calculate and log SAD (Sum of Absolute Differences).
 */
int calculate_and_log_sad(AVFilterContext *ctx, const AVFrame *frame, const AVFrame *prev_frame, int block_size, float *sad) {
    if (!sad) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for SAD.\n");
        return
         AVERROR(EINVAL);
    }

    if (frame->width != prev_frame->width || frame->height != prev_frame->height || frame->format != prev_frame->format) {
        av_log(ctx, AV_LOG_ERROR, "Frame format or size mismatch\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for SAD calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_frame_sad(frame, prev_frame, block_size, sad);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "SAD calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "SAD", *sad) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: SAD = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *sad, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Motion Vector Magnitude.
 */
int calculate_and_log_motion_vector_magnitude(AVFilterContext *ctx, const AVFrame *frame, double *motion_magnitude) {
    if (!motion_magnitude) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Motion Vector Magnitude.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Motion Vector Magnitude calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_motion_vector_magnitude(frame, motion_magnitude);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Motion vector magnitude calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Motion Vector Magnitude", *motion_magnitude) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Motion Vector Magnitude = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *motion_magnitude, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Edge Density.
 */
int calculate_and_log_edge_density(AVFilterContext *ctx, const AVFrame *frame, double *edge_density) {
    if (!edge_density) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Edge Density.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Edge Density calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_edge_density(frame, edge_density);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Edge density calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Edge Density", *edge_density) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Edge Density = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *edge_density, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Histogram Complexity.
 */
int calculate_and_log_histogram_complexity(AVFilterContext *ctx, const AVFrame *frame, double *histogram_complexity) {
    if (!histogram_complexity) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Histogram Complexity.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Histogram Complexity calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_histogram_complexity(frame, histogram_complexity);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Histogram complexity calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Histogram Complexity", *histogram_complexity) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Histogram Complexity = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *histogram_complexity, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Temporal Difference.
 */
int calculate_and_log_temporal_difference(AVFilterContext *ctx, const AVFrame *current_frame, const AVFrame *previous_frame, double *temporal_difference) {
    if (!temporal_difference) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Temporal Difference.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Temporal Difference calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_temporal_difference(current_frame, previous_frame, temporal_difference);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Temporal difference calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Temporal Difference", *temporal_difference) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (current_frame->pts != AV_NOPTS_VALUE) {
        time_sec = current_frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Temporal Difference = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *temporal_difference, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Color Variance.
 *
 * @param ctx The filter context for logging.
 * @param frame The input frame.
 * @param color_variance Pointer to store the calculated color_variance.
 * @return int 0 on success, non-zero on failure.
 */
int calculate_and_log_color_variance(AVFilterContext *ctx, const AVFrame *frame, double *color_variance) {
    if (!color_variance) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Color Variance.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Color Variance calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_color_variance(frame, color_variance);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Color Variance calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Color Variance", *color_variance) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Color Variance = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *color_variance, time_taken);
    return 0;
}

/**
 * @brief Calculate and log Texture Complexity.
 *
 * @param ctx The filter context for logging.
 * @param frame The input frame.
 * @param texture_complexity Pointer to store the calculated texture complexity.
 * @return int 0 on success, non-zero on failure.
 */
int calculate_and_log_texture_complexity(AVFilterContext *ctx, const AVFrame *frame, double *texture_complexity) {
    if (!texture_complexity) {
        av_log(ctx, AV_LOG_ERROR, "Invalid pointer for Texture Complexity.\n");
        return AVERROR(EINVAL);
    }

    // Start time measurement for Texture Complexity calculation
    clock_t start_time = clock();

    // Call the metric calculation function
    int ret = calculate_texture_complexity(frame, texture_complexity);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Texture complexity calculation failed.\n");
        return ret;
    }

    // End time measurement
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Validate the calculated value
    if (validate_metric(ctx, "Texture Complexity", *texture_complexity) < 0) {
        return AVERROR(EINVAL);
    }

    // Access the frame counter and timestamps
    CaeContext *cae_ctx = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Frame %d at %.2f sec: Texture Complexity = %.2f (Time taken: %.6f seconds)\n",
           cae_ctx->num_frames, time_sec, *texture_complexity, time_taken);
    return 0;
}

/**
 * @brief Helper function to validate metric values.
 *
 * @param ctx The filter context for logging.
 * @param metric_name Name of the metric being validated.
 * @param value The value of the metric to validate.
 * @return int 0 if valid, negative AVERROR code otherwise.
 */
int validate_metric(AVFilterContext *ctx, const char *metric_name, double value) {
    if (!isfinite(value)) {
        av_log(ctx, AV_LOG_WARNING, "%s calculation returned invalid value: %.2f\n", metric_name, value);
        return AVERROR(EINVAL);
    }
    return 0;
}

void normalize_metrics(FrameMetrics *metrics) {
    // Define maximum possible values for each metric based on their expected ranges
    const double MAX_DCT_ENERGY = 10000.0;
    const double MAX_SOBEL_ENERGY = 10000.0;
    const double MAX_SAD_ENERGY = 10000.0;
    const double MAX_MOTION_VECTOR_MAGNITUDE = 100.0;
    const double MAX_EDGE_DENSITY = 1.0;
    const double MAX_HISTOGRAM_COMPLEXITY = 1000.0;
    const double MAX_TEMPORAL_DIFFERENCE = 10000.0;
    const double MAX_TEXTURE_COMPLEXITY = 1.0;
    const double MAX_COLOR_VARIANCE = 1000.0;
    const double MAX_BLOCKINESS = 10.0;
    const double MAX_ENTROPY = 10.0;

    metrics->dct_energy = CLAMP_TO_UNIT(metrics->dct_energy / MAX_DCT_ENERGY);
    metrics->sobel_energy = CLAMP_TO_UNIT(metrics->sobel_energy / MAX_SOBEL_ENERGY);
    metrics->sad_energy = CLAMP_TO_UNIT(metrics->sad_energy / MAX_SAD_ENERGY);
    metrics->motion_vector_magnitude = CLAMP_TO_UNIT(metrics->motion_vector_magnitude / MAX_MOTION_VECTOR_MAGNITUDE);
    metrics->edge_density = CLAMP_TO_UNIT(metrics->edge_density / MAX_EDGE_DENSITY);
    metrics->histogram_complexity = CLAMP_TO_UNIT(metrics->histogram_complexity / MAX_HISTOGRAM_COMPLEXITY);
    metrics->temporal_difference = CLAMP_TO_UNIT(metrics->temporal_difference / MAX_TEMPORAL_DIFFERENCE);
    metrics->texture_complexity = CLAMP_TO_UNIT(metrics->texture_complexity / MAX_TEXTURE_COMPLEXITY);
    metrics->color_variance = CLAMP_TO_UNIT(metrics->color_variance / MAX_COLOR_VARIANCE);
    metrics->blockiness = CLAMP_TO_UNIT(metrics->blockiness / MAX_BLOCKINESS);
    metrics->entropy = CLAMP_TO_UNIT(metrics->entropy / MAX_ENTROPY);
}




