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
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <stddef.h> // For offsetof
#include "libavutil/internal.h"
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include "libavutil/opt.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"

#include <libavutil/avutil.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/frame.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <pthread.h> // For POSIX threads and mutex



static int attach_crf_metadata(AVFrame *frame, int crf);
static int predict_crf(CaeContext *ctx, FrameMetrics *metrics_window, int window_size);
static void predict_and_attach_crf(CaeContext *ctx, FrameMetrics *metrics_window, int window_size, int frame_num, AVFrame *frame);
static int setup_crf_and_vbv(CaeContext *ctx, AVFrame *frame, int crf);
static int init_cae_context(CaeContext *ctx);
static void uninit_cae_context(CaeContext *ctx);
static bool detect_scene_change(CaeContext *ctx, FrameMetrics *current_metrics, int frame_num, AVFrame *frame);
static av_cold int init_filter(AVFilterContext *ctx);
static av_cold void uninit_filter(AVFilterContext *ctx);
static int filter_frame(AVFilterLink *inlink, AVFrame *frame);
static double get_dynamic_weight(const char *metric, FrameMetrics *current_metrics);
static void cleanup_previous_frames(CaeContext *cae_ctx);
static AVFrame* get_previous_frame_with_format(CaeContext *cae_ctx, enum AVPixelFormat pix_fmt);
static int store_previous_frame(CaeContext *cae_ctx, AVFrame *current_frame);
double calculate_variance(FrameMetrics *metrics_window, int window_size, size_t offset);
void adjust_weights(CaeContext *ctx, FrameMetrics *metrics_window, int window_size);




/**
 * @brief Predict CRF based on frame metrics window.
 *
 * @param metrics_window Pointer to the metrics window.
 * @param window_size Size of the metrics window.
 * @return int Predicted CRF value (integer).
 */
static int predict_crf(CaeContext *ctx, FrameMetrics *metrics_window, int window_size) {
    // Calculate average complexity
    double total_complexity = 0.0;
    int num_metrics = ctx->metrics_window_filled ? window_size : ctx->metrics_window_index;
    for (int i = 0; i < num_metrics; i++) {
        FrameMetrics *metrics = &metrics_window[i];
        double complexity = metrics->dct_energy * ctx->weights_dct +
                            metrics->sobel_energy * ctx->weights_sobel +
                            metrics->sad_energy * ctx->weights_sad +
                            metrics->motion_vector_magnitude +
                            metrics->edge_density +
                            metrics->histogram_complexity +
                            metrics->temporal_difference +
                            metrics->texture_complexity +
                            metrics->color_variance +
                            metrics->blockiness +
                            metrics->entropy;
        total_complexity += complexity;
    }
    double average_complexity = total_complexity / num_metrics;

    // Normalize average_complexity between 0 and 1
    average_complexity = CLAMP_TO_UNIT(average_complexity);

    // Adjusted CRF prediction formula
    double predicted_crf = 28.0 - (25.0 / (1.0 + exp(-6.0 * (average_complexity - 0.5))));
    ctx->current_crf = (int)round(predicted_crf);
    return ctx->current_crf;
}

/**
 * @brief Attach CRF value to the frame's metadata.
 *
 * @param frame The AVFrame to which the metadata will be attached.
 * @param crf The predicted CRF value to attach.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static int attach_crf_metadata(AVFrame *frame, int crf) {
    char crf_value[16];
    snprintf(crf_value, sizeof(crf_value), "%d", crf);
    
    // Set metadata key "predicted_crf" to the CRF value
    int ret = av_dict_set(&frame->metadata, "predicted_crf", crf_value, 0);
    if (ret < 0) {
        return ret;
    }
    
    return 0;
}


/**
 * @brief Predict CRF based on metrics window and attach it to the frame's metadata.
 *
 * @param ctx The CAE context.
 * @param metrics_window Pointer to the metrics window.
 * @param window_size Size of the metrics window.
 * @param frame_num The actual frame number being processed.
 * @param frame The AVFrame being processed.
 */
static void predict_and_attach_crf(CaeContext *ctx, FrameMetrics *metrics_window, int window_size, int frame_num, AVFrame *frame) {
    // Normalize all metrics in the window
    for (int i = 0; i < window_size; i++) {
        normalize_metrics(&metrics_window[i]);
    }

    // Predict CRF based on metrics window
    int predicted_crf = predict_crf(ctx, metrics_window, window_size);
    av_log(ctx, AV_LOG_DEBUG, "Frame %d: Predicted CRF = %d\n", frame_num, predicted_crf);

    // Inject CRF and VBV settings into frame metadata
    int ret = setup_crf_and_vbv(ctx, frame, predicted_crf);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to inject CRF and VBV metadata at frame %d: %d\n", frame_num, ret);
    } else {
        av_log(ctx, AV_LOG_INFO, "CRF %d and VBV settings injected into frame %d metadata.\n", predicted_crf, frame_num);
    }
}

/**
 * @brief Initialize the CAE context with default or configured values.
 *
 * @param ctx Pointer to the CAE context.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static int init_cae_context(CaeContext *ctx) {
    // Initialize default values
    ctx->weights_dct = 0.2;
    ctx->weights_sobel = 0.1;
    ctx->weights_sad = 0.1;
    ctx->previous_complexity = 0.0;
    ctx->smoothed_complexity = 0.0;
    ctx->window_size = DEFAULT_WINDOW_SIZE;
    ctx->alpha = DEFAULT_ALPHA;
    ctx->beta = DEFAULT_BETA;
    ctx->lambda = DEFAULT_LAMBDA;
    ctx->window_index = 0;
    ctx->is_window_filled = false;
    ctx->gradual_change_count = 0;
    ctx->cumulative_complexity_change = 0.0;
    ctx->dct_block_size = DEFAULT_DCT_BLOCK_SIZE;
    ctx->sobel_block_size = DEFAULT_SOBEL_BLOCK_SIZE;
    ctx->sad_block_size = DEFAULT_SAD_BLOCK_SIZE;
    ctx->gradual_frame_count = DEFAULT_GRADUAL_FRAME_COUNT;
    ctx->gradual_threshold_multiplier = DEFAULT_GRADUAL_THRESHOLD_MULTIPLIER;
    ctx->num_frames = 0; // Initialize frame counter to 0

    // Initialize previous frames array
    for (int i = 0; i < MAX_PREV_FRAMES; i++) {
        ctx->prev_frames[i] = NULL;
    }
    ctx->num_prev_frames = 0;

    // Initialize dynamic weights (can be adjusted dynamically as needed)
    ctx->dynamic_weight_dct = ctx->weights_dct;
    ctx->dynamic_weight_sobel = ctx->weights_sobel;
    ctx->dynamic_weight_sad = ctx->weights_sad;

    // Validate window_size
    if (ctx->window_size < 1 || ctx->window_size > MAX_WINDOW_SIZE) {
        av_log(NULL, AV_LOG_ERROR, "Invalid window_size: %d. Must be between 1 and %d.\n", ctx->window_size, MAX_WINDOW_SIZE);
        return AVERROR(EINVAL);
    }

    // Validate weights
    double total_weights = ctx->weights_dct + ctx->weights_sobel + ctx->weights_sad;
    if (total_weights > 1.0) {
        av_log(NULL, AV_LOG_WARNING, "Sum of weights (DCT: %.2f, Sobel: %.2f, SAD: %.2f) exceeds 1.0. Normalizing weights.\n",
               ctx->weights_dct, ctx->weights_sobel, ctx->weights_sad);
        ctx->weights_dct /= total_weights;
        ctx->weights_sobel /= total_weights;
        ctx->weights_sad /= total_weights;
    }

    // Allocate memory for complexity_window based on window_size
    ctx->complexity_window = av_malloc_array(ctx->window_size, sizeof(double));
    if (!ctx->complexity_window) {
        av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory for complexity_window.\n");
        return AVERROR(ENOMEM);
    }

    // Initialize the complexity_window to zero
    memset(ctx->complexity_window, 0, ctx->window_size * sizeof(double));

    // Initialize metrics window
    ctx->metrics_window_size = DEFAULT_WINDOW_SIZE;
    ctx->metrics_window = av_malloc_array(ctx->metrics_window_size, sizeof(FrameMetrics));
    if (!ctx->metrics_window) {
        av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory for metrics_window.\n");
        av_freep(&ctx->complexity_window);
        return AVERROR(ENOMEM);
    }
    memset(ctx->metrics_window, 0, ctx->metrics_window_size * sizeof(FrameMetrics));
    ctx->metrics_window_index = 0;
    ctx->metrics_window_filled = false;

    // Initialize VBV parameters
    ctx->vbv_maxrate = 5000000;         // 5 Mbps
    ctx->vbv_bufsize = 10000000;        // 10 Mbps
    ctx->vbv_initial_buffer = 5000000;  // 5 Mbps (50% of bufsize)
    ctx->vbv_peakrate = 6000000;        // 6 Mbps
    ctx->vbv_minrate = 1000000;         // 1 Mbps
    ctx->vbv_adjust_cooldown = 0;
    ctx->vbv_cooldown_duration = 30;    // 30 frames cooldown

    av_log(NULL, AV_LOG_INFO, "CAE context initialized with window_size=%d, alpha=%.2f, beta=%.2f, lambda=%.2f, "
           "gradual_frame_count=%d, gradual_threshold_multiplier=%.2f, VBV_maxrate=%d, VBV_bufsize=%d\n",
           ctx->window_size, ctx->alpha, ctx->beta, ctx->lambda,
           ctx->gradual_frame_count, ctx->gradual_threshold_multiplier,
           ctx->vbv_maxrate, ctx->vbv_bufsize);

    return 0;
}


/**
 * @brief Uninitialize the CAE context by freeing allocated resources.
 *
 * @param ctx Pointer to the CAE context.
 */
static void uninit_cae_context(CaeContext *ctx) {
    if (ctx->complexity_window) {
        av_freep(&ctx->complexity_window);
    }
    if (ctx->metrics_window) {
        av_freep(&ctx->metrics_window);
    }
    av_log(NULL, AV_LOG_INFO, "CAE context uninitialized and resources freed.\n");
}

/**
 * @brief Detect scene changes based on frame metrics.
 *
 * @param ctx The CAE context.
 * @param metrics Pointer to the current frame's metrics.
 * @param frame_num The actual frame number being processed.
 * @param frame The AVFrame being processed.
 * @return bool True if a scene change is detected, False otherwise.
 */
static bool detect_scene_change(CaeContext *ctx, FrameMetrics *current_metrics, int frame_num, AVFrame *frame) {
    // Ensure that the metrics window is initialized
    if (!ctx->metrics_window || ctx->metrics_window_size <= 0) {
        av_log(ctx, AV_LOG_ERROR, "Metrics window is not initialized properly.\n");
        return false;
    }


    if (frame_num == 1) {
        return true;
    }

    // Step 1: Calculate the current complexity score using all metrics
    double weight_dct = get_dynamic_weight("dct_energy", current_metrics);
    double weight_sobel = get_dynamic_weight("sobel_energy", current_metrics);
    double weight_sad = get_dynamic_weight("sad_energy", current_metrics);
    double weight_motion = get_dynamic_weight("motion_vector_magnitude", current_metrics);
    double weight_edge = get_dynamic_weight("edge_density", current_metrics);
    double weight_histogram = get_dynamic_weight("histogram_complexity", current_metrics);
    double weight_temporal = get_dynamic_weight("temporal_difference", current_metrics);
    double weight_texture = get_dynamic_weight("texture_complexity", current_metrics);
    double weight_color = get_dynamic_weight("color_variance", current_metrics);
    double weight_blockiness = get_dynamic_weight("blockiness", current_metrics);
    double weight_entropy = get_dynamic_weight("entropy", current_metrics);

    // Compute the complexity score for the current frame
    double current_complexity = 
        (weight_dct * current_metrics->dct_energy) +
        (weight_sobel * current_metrics->sobel_energy) +
        (weight_sad * current_metrics->sad_energy) +
        (weight_motion * current_metrics->motion_vector_magnitude) +
        (weight_edge * current_metrics->edge_density) +
        (weight_histogram * current_metrics->histogram_complexity) +
        (weight_temporal * current_metrics->temporal_difference) +
        (weight_texture * current_metrics->texture_complexity) +
        (weight_color * current_metrics->color_variance) +
        (weight_blockiness * current_metrics->blockiness) +
        (weight_entropy * current_metrics->entropy);

    av_log(ctx, AV_LOG_DEBUG, "Frame %d: Current Complexity = %.2f\n", frame_num, current_complexity);

    // Step 2: Update the metrics window with the current complexity
    ctx->metrics_window[ctx->metrics_window_index].dct_energy = current_metrics->dct_energy;
    ctx->metrics_window[ctx->metrics_window_index].sobel_energy = current_metrics->sobel_energy;
    ctx->metrics_window[ctx->metrics_window_index].sad_energy = current_metrics->sad_energy;
    ctx->metrics_window[ctx->metrics_window_index].motion_vector_magnitude = current_metrics->motion_vector_magnitude;
    ctx->metrics_window[ctx->metrics_window_index].edge_density = current_metrics->edge_density;
    ctx->metrics_window[ctx->metrics_window_index].histogram_complexity = current_metrics->histogram_complexity;
    ctx->metrics_window[ctx->metrics_window_index].temporal_difference = current_metrics->temporal_difference;
    ctx->metrics_window[ctx->metrics_window_index].texture_complexity = current_metrics->texture_complexity;
    ctx->metrics_window[ctx->metrics_window_index].color_variance = current_metrics->color_variance;
    ctx->metrics_window[ctx->metrics_window_index].blockiness = current_metrics->blockiness;
    ctx->metrics_window[ctx->metrics_window_index].entropy = current_metrics->entropy;

    // Increment the window index and wrap around if necessary
    ctx->metrics_window_index = (ctx->metrics_window_index + 1) % ctx->metrics_window_size;
    if (ctx->metrics_window_index == 0) {
        ctx->metrics_window_filled = true;
    }

    // Step 3: Calculate moving average and standard deviation if the window is filled
    double moving_average = 0.0;
    double variance = 0.0;
    double stddev = 0.0;
    double threshold = 0.0;

    if (ctx->metrics_window_filled) {
        // Calculate the sum of complexities in the window
        for (int i = 0; i < ctx->metrics_window_size; i++) {
            double frame_complexity = 
                (weight_dct * ctx->metrics_window[i].dct_energy) +
                (weight_sobel * ctx->metrics_window[i].sobel_energy) +
                (weight_sad * ctx->metrics_window[i].sad_energy) +
                (weight_motion * ctx->metrics_window[i].motion_vector_magnitude) +
                (weight_edge * ctx->metrics_window[i].edge_density) +
                (weight_histogram * ctx->metrics_window[i].histogram_complexity) +
                (weight_temporal * ctx->metrics_window[i].temporal_difference) +
                (weight_texture * ctx->metrics_window[i].texture_complexity) +
                (weight_color * ctx->metrics_window[i].color_variance) +
                (weight_blockiness * ctx->metrics_window[i].blockiness) +
                (weight_entropy * ctx->metrics_window[i].entropy);
            moving_average += frame_complexity;
        }
        moving_average /= ctx->metrics_window_size;

        // Calculate variance
        for (int i = 0; i < ctx->metrics_window_size; i++) {
            double frame_complexity = 
                (weight_dct * ctx->metrics_window[i].dct_energy) +
                (weight_sobel * ctx->metrics_window[i].sobel_energy) +
                (weight_sad * ctx->metrics_window[i].sad_energy) +
                (weight_motion * ctx->metrics_window[i].motion_vector_magnitude) +
                (weight_edge * ctx->metrics_window[i].edge_density) +
                (weight_histogram * ctx->metrics_window[i].histogram_complexity) +
                (weight_temporal * ctx->metrics_window[i].temporal_difference) +
                (weight_texture * ctx->metrics_window[i].texture_complexity) +
                (weight_color * ctx->metrics_window[i].color_variance) +
                (weight_blockiness * ctx->metrics_window[i].blockiness) +
                (weight_entropy * ctx->metrics_window[i].entropy);
            variance += pow(frame_complexity - moving_average, 2);
        }
        variance /= ctx->metrics_window_size;
        stddev = sqrt(variance);

        av_log(ctx, AV_LOG_DEBUG, "Frame %d: Moving Average = %.2f, Stddev = %.2f\n", frame_num, moving_average, stddev);

        // Step 4: Calculate the dynamic threshold
        threshold = (ctx->alpha * moving_average) + (0.25 * stddev) + ctx->beta;
        av_log(ctx, AV_LOG_DEBUG, "Frame %d: Threshold = %.2f\n", frame_num, threshold);
    }

    // Step 5: Smooth the complexity using Exponential Moving Average (EMA)
    ctx->smoothed_complexity = (ctx->lambda * current_complexity) + ((1.0 - ctx->lambda) * ctx->smoothed_complexity);
    av_log(ctx, AV_LOG_DEBUG, "Frame %d: Smoothed Complexity = %.2f\n", frame_num, ctx->smoothed_complexity);

    // Step 6: Calculate the change in complexity
    double delta_complexity = fabs(current_complexity - ctx->previous_complexity);
    av_log(ctx, AV_LOG_DEBUG, "Frame %d: Delta Complexity = %.2f\n", frame_num, delta_complexity);

    // Step 7: Scene Change Detection Logic
    bool scene_change_detected = false;

    if (ctx->metrics_window_filled) {
        if (delta_complexity > threshold) {
            // **Abrupt Scene Change Detected**
            av_log(ctx, AV_LOG_INFO, "Scene change detected at frame %d (Delta Complexity: %.2f > Threshold: %.2f)\n", frame_num, delta_complexity, threshold);
            scene_change_detected = true;

            // Reset gradual change tracking variables
            ctx->gradual_change_count = 0;
            ctx->cumulative_complexity_change = 0.0;
        } else if (delta_complexity > (threshold * 0.8)) {
            // **Potential Gradual Scene Change**
            ctx->gradual_change_count++;
            ctx->cumulative_complexity_change += delta_complexity;
            av_log(ctx, AV_LOG_DEBUG, "Frame %d: Gradual change count = %d, Cumulative Change = %.2f\n", frame_num, ctx->gradual_change_count, ctx->cumulative_complexity_change);

            if (ctx->gradual_change_count >= ctx->gradual_frame_count &&
                ctx->cumulative_complexity_change > (threshold * ctx->gradual_threshold_multiplier)) {
                av_log(ctx, AV_LOG_INFO, "Gradual scene change detected at frame %d\n", frame_num);
                scene_change_detected = true;

                // Reset gradual change tracking variables
                ctx->gradual_change_count = 0;
                ctx->cumulative_complexity_change = 0.0;
            }
        } else {
            // **No Significant Change Detected**
            // Optionally, implement decay mechanism for gradual change tracking
            if (ctx->gradual_change_count > 0) {
                ctx->gradual_change_count--;
                ctx->cumulative_complexity_change *= 0.9; // Apply decay to cumulative change
                if (ctx->gradual_change_count < 0) {
                    ctx->gradual_change_count = 0;
                    ctx->cumulative_complexity_change = 0.0;
                }
                av_log(ctx, AV_LOG_DEBUG, "Frame %d: Gradual change count decayed to %d, Cumulative Change = %.2f\n", frame_num, ctx->gradual_change_count, ctx->cumulative_complexity_change);
            }
        }
    }

    // Step 8: Update the previous complexity for the next frame's comparison
    ctx->previous_complexity = current_complexity;

    return scene_change_detected;
}

/**
 * @brief Initialize the CAE filter.
 *
 * @param ctx The filter context.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static av_cold int init_filter(AVFilterContext *ctx) {
    CaeContext *cae_ctx = ctx->priv;
    // Initialize the mutex
    pthread_mutex_init(&cae_ctx->mutex, NULL);
    int ret = init_cae_context(cae_ctx);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize CAE context.\n");
        return ret;
    }

    // Open the CSV file for writing
    cae_ctx->metrics_file = fopen(cae_ctx->metrics_filename, "w");
    if (!cae_ctx->metrics_file) {
        av_log(ctx, AV_LOG_ERROR, "Failed to open metrics file: %s\n", cae_ctx->metrics_filename);
        return AVERROR(EIO);
    }

    // Write the CSV header
    fprintf(cae_ctx->metrics_file, "Frame,DCTEnergy,SobelEnergy,SADEnergy,MotionVectorMagnitude,EdgeDensity,HistogramComplexity,TemporalDifference,TextureComplexity,ColorVariance,Blockiness,Entropy\n");


    av_log(ctx, AV_LOG_INFO, "CAE filter initialized successfully.\n");
    return 0;
}

/**
 * @brief Uninitialize the CAE filter.
 *
 * @param ctx The filter context.
 */
static av_cold void uninit_filter(AVFilterContext *ctx) {
    CaeContext *cae_ctx = ctx->priv;
    cleanup_previous_frames(cae_ctx);
    uninit_cae_context(cae_ctx);

    // Close the CSV file
    if (cae_ctx->metrics_file) {
        fclose(cae_ctx->metrics_file);
        cae_ctx->metrics_file = NULL;
    }

    // Destroy the mutex
    pthread_mutex_destroy(&cae_ctx->mutex);

    av_log(ctx, AV_LOG_INFO, "CAE filter uninitialized successfully.\n");
}

// Main Filter Frame Function
static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
    AVFilterContext *ctx = inlink->dst;
    CaeContext *cae_ctx = ctx->priv;
    int ret;

    // Start overall frame processing time measurement
    clock_t frame_start_time = clock();

    av_log(ctx, AV_LOG_DEBUG, "Incoming frame picture type = %d\n", frame->pict_type);

    // Increment frame counter
    cae_ctx->num_frames++;

    // Access frame timestamps
    double time_sec = 0.0;
    if (frame->pts != AV_NOPTS_VALUE) {
        time_sec = frame->pts * av_q2d(inlink->time_base);
    }

    av_log(ctx, AV_LOG_DEBUG, "Processing frame %d at %.2f sec.\n", cae_ctx->num_frames, time_sec);

    // Initialize FrameMetrics
    FrameMetrics metrics;
    memset(&metrics, 0, sizeof(FrameMetrics));

    // Calculate essential metrics
    ret = calculate_and_log_dct_energy(ctx, frame, cae_ctx->dct_block_size, &metrics.dct_energy);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Skipping scene change detection due to DCT energy calculation failure at frame %d.\n", cae_ctx->num_frames);
        return ff_filter_frame(ctx->outputs[0], frame);
    }

    ret = calculate_and_log_sobel_energy(ctx, frame, cae_ctx->sobel_block_size, &metrics.sobel_energy);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Skipping scene change detection due to Sobel energy calculation failure at frame %d.\n", cae_ctx->num_frames);
        return ff_filter_frame(ctx->outputs[0], frame);
    }

    // Retrieve a previous frame with the same format for SAD calculation
    AVFrame *prev_frame = get_previous_frame_with_format(cae_ctx, frame->format);
    if (!prev_frame) {
        av_log(ctx, AV_LOG_INFO, "No previous frame with matching format found. Assigning default values for dependent metrics.\n");
        metrics.sad_energy = 0.0;
        metrics.temporal_difference = 0.0;
    } else {
        // Perform SAD calculation between current frame and the previous matching frame
        ret = calculate_and_log_sad(ctx, frame, prev_frame, cae_ctx->sad_block_size, &metrics.sad_energy);
        if (ret < 0) {
            metrics.sad_energy = 0.0;
            av_log(ctx, AV_LOG_ERROR, "SAD calculation failed.\n");
        }

        // Calculate temporal difference
        ret = calculate_and_log_temporal_difference(ctx, frame, prev_frame, &metrics.temporal_difference);
        if (ret != 0) {
            av_log(ctx, AV_LOG_WARNING, "Temporal Difference calculation failed at frame %d.\n", cae_ctx->num_frames);
            metrics.temporal_difference = 0.0;
        }
    }

    // Compute additional metrics for all frames
    ret = calculate_and_log_motion_vector_magnitude(ctx, frame, &metrics.motion_vector_magnitude);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Motion Vector Magnitude calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.motion_vector_magnitude = 0.0;
    }

    ret = calculate_and_log_edge_density(ctx, frame, &metrics.edge_density);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Edge Density calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.edge_density = 0.0;
    }

    ret = calculate_and_log_histogram_complexity(ctx, frame, &metrics.histogram_complexity);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Histogram Complexity calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.histogram_complexity = 0.0;
    }

    ret = calculate_and_log_texture_complexity(ctx, frame, &metrics.texture_complexity);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Texture Complexity calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.texture_complexity = 0.0;
    }

    ret = calculate_and_log_color_variance(ctx, frame, &metrics.color_variance);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Color Variance calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.color_variance = 0.0;
    }

    ret = calculate_and_log_blockiness(ctx, frame, &metrics.blockiness);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Blockiness calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.blockiness = 0.0;
    }

    ret = calculate_and_log_entropy(ctx, frame, &metrics.entropy);
    if (ret != 0) {
        av_log(ctx, AV_LOG_WARNING, "Entropy calculation failed at frame %d.\n", cae_ctx->num_frames);
        metrics.entropy = 0.0;
    }

    pthread_mutex_lock(&cae_ctx->mutex);
    fprintf(cae_ctx->metrics_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            cae_ctx->num_frames,
            metrics.dct_energy,
            metrics.sobel_energy,
            metrics.sad_energy,
            metrics.motion_vector_magnitude,
            metrics.edge_density,
            metrics.histogram_complexity,
            metrics.temporal_difference,
            metrics.texture_complexity,
            metrics.color_variance,
            metrics.blockiness,
            metrics.entropy);
    pthread_mutex_unlock(&cae_ctx->mutex);

    // Update metrics window with all metrics
    cae_ctx->metrics_window[cae_ctx->metrics_window_index] = metrics;
    cae_ctx->metrics_window_index = (cae_ctx->metrics_window_index + 1) % cae_ctx->metrics_window_size;
    if (cae_ctx->metrics_window_index == 0) {
        cae_ctx->metrics_window_filled = true;
    }

    av_log(ctx, AV_LOG_DEBUG, "metrics_window[%d] updated.\n", cae_ctx->metrics_window_index);

    // Detect Scene Changes
    bool scene_change = detect_scene_change(cae_ctx, &metrics, cae_ctx->num_frames, frame);

    if (scene_change) {
        av_log(ctx, AV_LOG_INFO, "Scene change detected at frame %d.\n",cae_ctx->num_frames);
        // Predict and attach CRF and VBV metadata
        //predict_and_attach_crf(cae_ctx, cae_ctx->metrics_window, cae_ctx->metrics_window_size, cae_ctx->num_frames, frame);
    }

    // Store the current frame in the previous frames buffer for future reference
    ret = store_previous_frame(cae_ctx, frame);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to store previous frame.\n");
    }

    // End overall frame processing time measurement
    clock_t frame_end_time = clock();
    double frame_processing_time = ((double)(frame_end_time - frame_start_time)) / CLOCKS_PER_SEC;
    cae_ctx->processing_time_sec = frame_processing_time;

    av_log(ctx, AV_LOG_DEBUG, "Frame %d processed in %.6f seconds.\n", cae_ctx->num_frames, cae_ctx->processing_time_sec);


    // Push the Frame to the Next Filter
    ret = ff_filter_frame(ctx->outputs[0], frame);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push frame to next filter at frame %d: %d\n", cae_ctx->num_frames, ret);
    }

    return ret;
}

// Dynamic weight calculation based on metric variance
void adjust_weights(CaeContext *ctx, FrameMetrics *metrics_window, int window_size) {
    double variance_dct = calculate_variance(metrics_window, window_size, offsetof(FrameMetrics, dct_energy));
    double variance_sad = calculate_variance(metrics_window, window_size, offsetof(FrameMetrics, sad_energy));
    double variance_edge = calculate_variance(metrics_window, window_size, offsetof(FrameMetrics, edge_density));

    // Calculate total variance for normalization
    double total_variance = variance_dct + variance_sad + variance_edge;

    if (total_variance > 0) {
        // Adjust weights inversely proportional to variance
        ctx->weights_dct = (1.0 - variance_dct / total_variance);
        ctx->weights_sad = (1.0 - variance_sad / total_variance);
        ctx->weights_edge = (1.0 - variance_edge / total_variance);
    }
}

// Helper function to calculate variance for a specific metric
double calculate_variance(FrameMetrics *metrics_window, int window_size, size_t offset) {
    double mean = 0.0, variance = 0.0;

    for (int i = 0; i < window_size; i++) {
        double value = *(double *)((uint8_t *)&metrics_window[i] + offset);
        mean += value;
    }
    mean /= window_size;

    for (int i = 0; i < window_size; i++) {
        double value = *(double *)((uint8_t *)&metrics_window[i] + offset);
        variance += (value - mean) * (value - mean);
    }
    return variance / window_size;
}


/**
 * @brief Inject CRF and VBV settings into the frame's metadata.
 *
 * @param ctx The CAE context.
 * @param frame The AVFrame to which metadata will be attached.
 * @param crf The predicted CRF value.
 * @return int 0 on success, negative AVERROR code on failure.
 */
static int setup_crf_and_vbv(CaeContext *ctx, AVFrame *frame, int crf) {
    int ret;
    
    // Convert integer values to strings for metadata
    char crf_str[16];
    char vbv_maxrate_str[32];
    char vbv_bufsize_str[32];
    char vbv_initial_buffer_str[32];
    char vbv_peakrate_str[32];
    char vbv_minrate_str[32];
    
    snprintf(crf_str, sizeof(crf_str), "%d", crf);
    snprintf(vbv_maxrate_str, sizeof(vbv_maxrate_str), "%d", ctx->vbv_maxrate);
    snprintf(vbv_bufsize_str, sizeof(vbv_bufsize_str), "%d", ctx->vbv_bufsize);
    snprintf(vbv_initial_buffer_str, sizeof(vbv_initial_buffer_str), "%d", ctx->vbv_initial_buffer);
    snprintf(vbv_peakrate_str, sizeof(vbv_peakrate_str), "%d", ctx->vbv_peakrate);
    snprintf(vbv_minrate_str, sizeof(vbv_minrate_str), "%d", ctx->vbv_minrate);
    
    // Attach CRF metadata
    ret = av_dict_set(&frame->metadata, METADATA_KEY_PREDICTED_CRF, crf_str, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to set %s metadata: %d\n", METADATA_KEY_PREDICTED_CRF, ret);
        return ret;
    }
    
    // Attach VBV metadata
    ret = av_dict_set(&frame->metadata, METADATA_KEY_VBV_MAXRATE, vbv_maxrate_str, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to set %s metadata: %d\n", METADATA_KEY_VBV_MAXRATE, ret);
        return ret;
    }
    
    ret = av_dict_set(&frame->metadata, METADATA_KEY_VBV_BUFSIZE, vbv_bufsize_str, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to set %s metadata: %d\n", METADATA_KEY_VBV_BUFSIZE, ret);
        return ret;
    }
    
    ret = av_dict_set(&frame->metadata, METADATA_KEY_VBV_INITIAL_BUFFER, vbv_initial_buffer_str, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to set %s metadata: %d\n", METADATA_KEY_VBV_INITIAL_BUFFER, ret);
        return ret;
    }
    
    ret = av_dict_set(&frame->metadata, METADATA_KEY_VBV_PEAKRATE, vbv_peakrate_str, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to set %s metadata: %d\n", METADATA_KEY_VBV_PEAKRATE, ret);
        return ret;
    }
    
    ret = av_dict_set(&frame->metadata, METADATA_KEY_VBV_MINRATE, vbv_minrate_str, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to set %s metadata: %d\n", METADATA_KEY_VBV_MINRATE, ret);
        return ret;
    }
    
    av_log(ctx, AV_LOG_INFO, "Injected CRF and VBV settings into frame metadata.\n");
    
    return 0;
}

/**
 * @brief Retrieve dynamic weight based on metric name and current metrics.
 *
 * @param metric Metric name.
 * @param current_metrics Pointer to current frame's metrics.
 * @return double Dynamic weight for the specified metric.
 */
static double get_dynamic_weight(const char *metric, FrameMetrics *current_metrics) {
    // Dynamic weight adjustments based on metric thresholds
    if (strcmp(metric, "motion_vector_magnitude") == 0) {
        if (current_metrics->motion_vector_magnitude > 0.75) {
            return 0.2; // High motion
        } else if (current_metrics->motion_vector_magnitude < 0.25) {
            return 0.05; // Low motion
        }
    }

    if (strcmp(metric, "edge_density") == 0) {
        if (current_metrics->edge_density > 0.7) {
            return 0.15;
        } else if (current_metrics->edge_density < 0.3) {
            return 0.05;
        }
    }

    if (strcmp(metric, "histogram_complexity") == 0) {
        if (current_metrics->histogram_complexity > 50.0) {
            return 0.15;
        } else if (current_metrics->histogram_complexity < 20.0) {
            return 0.05;
        }
    }

    if (strcmp(metric, "temporal_difference") == 0) {
        if (current_metrics->temporal_difference > 50.0) {
            return 0.15;
        } else if (current_metrics->temporal_difference < 10.0) {
            return 0.05;
        }
    }

    if (strcmp(metric, "texture_complexity") == 0) {
        if (current_metrics->texture_complexity > 1.0) {
            return 0.15;
        } else if (current_metrics->texture_complexity < 0.3) {
            return 0.05;
        }
    }

    // Add dynamic conditions for other metrics as needed

    // Default weights
    if (strcmp(metric, "dct_energy") == 0) return 0.1;
    if (strcmp(metric, "sobel_energy") == 0) return 0.05;
    if (strcmp(metric, "sad_energy") == 0) return 0.05;
    if (strcmp(metric, "color_variance") == 0) return 0.05;
    if (strcmp(metric, "blockiness") == 0) return 0.05;
    if (strcmp(metric, "entropy") == 0) return 0.05;

    return 0.0;
}

static int store_previous_frame(CaeContext *cae_ctx, AVFrame *current_frame) {
    AVFrame *cloned_frame = av_frame_clone(current_frame);
    if (!cloned_frame) {
        av_log(NULL, AV_LOG_ERROR, "Failed to clone frame for previous frame storage.\n");
        return AVERROR(ENOMEM);
    }

    // If the buffer is full, free and remove the oldest frame
    if (cae_ctx->num_prev_frames == MAX_PREV_FRAMES) {
        av_frame_free(&cae_ctx->prev_frames[0]);
        // Shift frames in the buffer to the left
        for (int i = 1; i < MAX_PREV_FRAMES; i++) {
            cae_ctx->prev_frames[i - 1] = cae_ctx->prev_frames[i];
        }
        cae_ctx->num_prev_frames--;
    }

    // Store the cloned frame
    cae_ctx->prev_frames[cae_ctx->num_prev_frames] = cloned_frame;
    cae_ctx->num_prev_frames++;

    return 0;
}

static AVFrame* get_previous_frame_with_format(CaeContext *cae_ctx, enum AVPixelFormat pix_fmt) {
    for (int i = cae_ctx->num_prev_frames - 1; i >= 0; i--) {
        if (cae_ctx->prev_frames[i] && cae_ctx->prev_frames[i]->format == pix_fmt) {
            return cae_ctx->prev_frames[i];
        }
    }
    // No matching frame found
    return NULL;
}

static void cleanup_previous_frames(CaeContext *cae_ctx) {
    for (int i = 0; i < MAX_PREV_FRAMES; i++) {
        if (cae_ctx->prev_frames[i]) {
            av_frame_free(&cae_ctx->prev_frames[i]);
        }
    }
    cae_ctx->num_prev_frames = 0;
}

/**
 * @brief Define filter options (configurable parameters).
 */
// Filter Options
static const AVOption cae_options[] = {
    { "window_size", "Size of the moving window for complexity averaging",
      offsetof(CaeContext, window_size), AV_OPT_TYPE_INT,
      {.i64 = DEFAULT_WINDOW_SIZE}, 1, MAX_WINDOW_SIZE, FLAGS },
    { "alpha", "Alpha multiplier for threshold calculation",
      offsetof(CaeContext, alpha), AV_OPT_TYPE_DOUBLE,
      {.dbl = DEFAULT_ALPHA}, 0.0, 10.0, FLAGS },
    { "beta", "Beta addition for threshold calculation",
      offsetof(CaeContext, beta), AV_OPT_TYPE_DOUBLE,
      {.dbl = DEFAULT_BETA}, 0.0, 100.0, FLAGS },
    { "lambda", "Lambda for EMA smoothing",
      offsetof(CaeContext, lambda), AV_OPT_TYPE_DOUBLE,
      {.dbl = DEFAULT_LAMBDA}, 0.0, 1.0, FLAGS },
    { "weight_dct", "Weight for DCT energy",
      offsetof(CaeContext, weights_dct), AV_OPT_TYPE_DOUBLE,
      {.dbl = 0.1}, 0.0, 1.0, FLAGS },
    { "weight_sobel", "Weight for Sobel energy",
      offsetof(CaeContext, weights_sobel), AV_OPT_TYPE_DOUBLE,
      {.dbl = 0.05}, 0.0, 1.0, FLAGS },
    { "weight_sad", "Weight for SAD energy",
      offsetof(CaeContext, weights_sad), AV_OPT_TYPE_DOUBLE,
      {.dbl = 0.05}, 0.0, 1.0, FLAGS },
    { "gradual_frame_count", "Number of consecutive frames for gradual scene change detection",
      offsetof(CaeContext, gradual_frame_count), AV_OPT_TYPE_INT,
      {.i64 = DEFAULT_GRADUAL_FRAME_COUNT}, 1, 100, FLAGS },
    { "gradual_threshold_multiplier", "Multiplier for cumulative complexity change to detect gradual scene changes",
      offsetof(CaeContext, gradual_threshold_multiplier), AV_OPT_TYPE_DOUBLE,
      {.dbl = DEFAULT_GRADUAL_THRESHOLD_MULTIPLIER}, 0.0, 10.0, FLAGS },
    { "dct_block_size", "Block size for DCT energy calculation",
      offsetof(CaeContext, dct_block_size), AV_OPT_TYPE_INT,
      {.i64 = DEFAULT_DCT_BLOCK_SIZE}, 4, 16, FLAGS },
    { "sobel_block_size", "Block size for Sobel energy calculation",
      offsetof(CaeContext, sobel_block_size), AV_OPT_TYPE_INT,
      {.i64 = DEFAULT_SOBEL_BLOCK_SIZE}, 3, 16, FLAGS },
    { "sad_block_size", "Block size for SAD calculation",
      offsetof(CaeContext, sad_block_size), AV_OPT_TYPE_INT,
      {.i64 = DEFAULT_SAD_BLOCK_SIZE}, 4, 16, FLAGS },
    { "vbv_maxrate", "Maximum VBV bitrate (bps)",
      offsetof(CaeContext, vbv_maxrate), AV_OPT_TYPE_INT,
      {.i64 = 5000000}, 100000, 100000000, FLAGS },
    { "vbv_bufsize", "VBV buffer size (bps)",
      offsetof(CaeContext, vbv_bufsize), AV_OPT_TYPE_INT,
      {.i64 = 10000000}, 100000, 200000000, FLAGS },
    { "vbv_initial_buffer", "Initial VBV buffer fullness (bps)",
      offsetof(CaeContext, vbv_initial_buffer), AV_OPT_TYPE_INT,
      {.i64 = 5000000}, 100000, 200000000, FLAGS },
    { "vbv_peakrate", "VBV peak bitrate (bps)",
      offsetof(CaeContext, vbv_peakrate), AV_OPT_TYPE_INT,
      {.i64 = 6000000}, 100000, 200000000, FLAGS },
    { "vbv_minrate", "VBV minimum bitrate (bps)",
      offsetof(CaeContext, vbv_minrate), AV_OPT_TYPE_INT,
      {.i64 = 1000000}, 50000, 100000000, FLAGS },
    { "metrics_filename", "Filename for metrics CSV output", offsetof(CaeContext, metrics_filename), AV_OPT_TYPE_STRING, { .str = "metrics.csv" }, 0, 0, FLAGS },
  
    { NULL }
};

/**
 * @brief Define supported pixel formats.
 */
const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P,
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
    }
};

/**
 * @brief Register the CAE filter with FFmpeg.
 */
const AVFilter ff_vf_cae = {
    .name          = "cae",
    .description   = NULL_IF_CONFIG_SMALL("Detect scene changes based on frame metrics (DCT, Sobel, SAD)."),
    .priv_size     = sizeof(CaeContext),
    .priv_class    = &cae_class,
    .init          = init_filter,
    .uninit        = uninit_filter,
    .flags         = AVFILTER_FLAG_METADATA_ONLY,
    FILTER_INPUTS(avfilter_vf_cae_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
