/**
 * @file calculate_frame_sad.c
 * @brief Calculates the Sum of Absolute Differences (SAD) between two AVFrames with support for AVX-512, NEON, and Scalar processing.
 *
 * This implementation dynamically selects the best available optimization (AVX-512, NEON, or Scalar)
 * based on the CPU's capabilities at runtime. It ensures optimal performance across diverse hardware
 * platforms while maintaining robustness through comprehensive error handling.
 *
 * Author: Zaki Ahmed
 * Date: 2024-05-14
 */

#include <libavutil/frame.h>       // For AVFrame
#include <libavutil/imgutils.h>    // For av_image_alloc
#include <libavutil/mem.h>         // For av_freep
#include <omp.h>                   // For OpenMP parallelization
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>             // For AVX-512 intrinsics
#include <cpuid.h>                 // For __get_cpuid_count
#endif
#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>              // For NEON intrinsics
#endif
#include <stdlib.h>                // For aligned_alloc, free
#include <stdio.h>                 // For fprintf, stderr, printf
#include <string.h>                // For memset
#include <stdint.h>                // For uint8_t
#include <stdbool.h>               // For bool type
#include <math.h>                  // For fabsf

/* ----------------------------
 * Error Codes
 * ----------------------------
 *
 * Define all error codes as static const integers to ensure they have internal linkage
 * and are confined to this source file.
 */
static const int ERR_NULL_FRAME_A          = -1;
static const int ERR_NULL_FRAME_B          = -2;
static const int ERR_UNSUPPORTED_PIX_FMT   = -3;
static const int ERR_INVALID_PIX_FMT       = -4;
static const int ERR_MEMORY_ALLOCATION     = -5;
static const int ERR_PARALLEL_CANCELLATION = -6;
static const int ERR_ZERO_AREA             = -7;
static const int ERR_NULL_OUTPUT           = -8;
// Add more error codes as needed

/* ----------------------------
 * Structure Definitions
 * ----------------------------
 *
 * Define the SAD_Functions structure only once to avoid redefinition conflicts.
 */
typedef struct SAD_Functions {
    /**
     * @brief Function pointer for block extraction.
     *
     * Extracts corresponding blocks from two AVFrames.
     *
     * @param frame_a Pointer to the first AVFrame (reference frame).
     * @param frame_b Pointer to the second AVFrame (current frame).
     * @param block_a Pointer to the sad_block_size x sad_block_size block buffer from frame A.
     * @param block_b Pointer to the sad_block_size x sad_block_size block buffer from frame B.
     * @param x The x-coordinate of the top-left corner of the block.
     * @param y The y-coordinate of the top-left corner of the block.
     * @param plane The plane index (0 for Y, 1 for U, 2 for V).
     * @param sad_block_size The size of the SAD block.
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*extract_block)(const AVFrame *frame_a, const AVFrame *frame_b, float *block_a, float *block_b, int x, int y, int plane, int sad_block_size);
    
    /**
     * @brief Function pointer for calculating the Sum of Absolute Differences (SAD).
     *
     * Calculates the SAD between two blocks.
     *
     * @param block_a Pointer to the first block data.
     * @param block_b Pointer to the second block data.
     * @param sad Pointer to store the calculated SAD.
     * @param sad_block_size The size of the SAD block.
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*calculate_sad)(const float *block_a, const float *block_b, float *sad, int sad_block_size);
} SAD_Functions;

/* ----------------------------
 * Utility Macros
 * ----------------------------
 */
/**
 * @brief Macro to abstract aligned memory allocation for portability.
 */
#if defined(_MSC_VER)
    #include <malloc.h> // For _aligned_malloc and _aligned_free
    #define ALIGN_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGN_FREE(ptr) _aligned_free(ptr)
#elif defined(__GNUC__) || defined(__clang__)
    #define ALIGN_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGN_FREE(ptr) free(ptr)
#else
    #error "Aligned memory allocation not supported on this compiler."
#endif

/**
 * @def CACHE_LINE_SIZE
 * @brief Defines the cache line size in bytes based on the target architecture.
 */
#ifndef CACHE_LINE_SIZE
    #if defined(__x86_64__) || defined(_M_X64)
        #define CACHE_LINE_SIZE 64 // Typical for x86-64
    #elif defined(__aarch64__) || defined(__arm__)
        #define CACHE_LINE_SIZE 32 // Typical for ARM
    #else
        #define CACHE_LINE_SIZE 64 // Default fallback
    #endif
#endif

/* ----------------------------
 * Internal Function Declarations
 * ----------------------------
 *
 * These functions are intended to be used only within this source file and are not exposed externally.
 * Hence, they are declared as static.
 */

/* Scalar Implementations */
static int extract_sad_block_scalar(const AVFrame *frame_a, const AVFrame *frame_b,
                                    float *block_a, float *block_b, int x, int y,
                                    int plane, int sad_block_size);
static int calculate_sad_scalar(const float *block_a, const float *block_b, float *sad, int sad_block_size);

/* NEON Implementations */
#if defined(__arm__) || defined(__aarch64__)
static int extract_sad_blocks_from_frame_neon(const AVFrame *frame_a, const AVFrame *frame_b,
                                             float *block_a, float *block_b,
                                             int x, int y, int plane, int sad_block_size);
static int calculate_sad_neon(const float *block_a, const float *block_b, float *sad, int sad_block_size);
#endif

/* AVX-512 Implementations */
#if defined(__AVX512F__)
static int extract_sad_block_avx512(const AVFrame *frame_a, const AVFrame *frame_b,
                                    float *block_a, float *block_b, int x, int y,
                                    int plane, int sad_block_size);
static int calculate_sad_avx512(const float *block_a, const float *block_b, float *sad, int sad_block_size);
#endif

/* ----------------------------
 * Internal Utility Functions
 * ----------------------------
 */

/**
 * @brief Logs an error message based on the error code.
 *
 * @param error_code The error code indicating the type of error.
 */
static void log_error(int error_code) {
    switch (error_code) {
        case ERR_NULL_FRAME_A:
            fprintf(stderr, "Error: NULL frame A pointer.\n");
            break;
        case ERR_NULL_FRAME_B:
            fprintf(stderr, "Error: NULL frame B pointer.\n");
            break;
        case ERR_UNSUPPORTED_PIX_FMT:
            fprintf(stderr, "Error: Unsupported pixel format.\n");
            break;
        case ERR_INVALID_PIX_FMT:
            fprintf(stderr, "Error: Invalid pixel format or mismatched formats.\n");
            break;
        case ERR_MEMORY_ALLOCATION:
            fprintf(stderr, "Error: Memory allocation failed.\n");
            break;
        case ERR_PARALLEL_CANCELLATION:
            fprintf(stderr, "Error: Parallel processing was cancelled due to a critical error.\n");
            break;
        case ERR_ZERO_AREA:
            fprintf(stderr, "Error: Frame area is zero, cannot normalize SAD.\n");
            break;
        case ERR_NULL_OUTPUT:
            fprintf(stderr, "Error: NULL output SAD pointer.\n");
            break;
        default:
            fprintf(stderr, "Error: Unknown error code %d.\n", error_code);
            break;
    }
}

/**
 * @brief Retrieves the weight for a given plane based on the pixel format.
 *
 * @param pix_fmt The pixel format of the frame.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @return double The weight corresponding to the plane.
 */
static double get_plane_weight(enum AVPixelFormat pix_fmt, int plane) {
    switch (pix_fmt) {
        case AV_PIX_FMT_YUV420P:
            return (plane == 0) ? 1.0 : 0.5;  // Y:1.0, U/V:0.5
        case AV_PIX_FMT_YUV422P:
            return (plane == 0) ? 1.0 : 0.75; // Y:1.0, U/V:0.75
        case AV_PIX_FMT_YUV444P:
            return 1.0; // Y/U/V:1.0
        default:
            // Default weight; should not reach here if pixel format is validated beforehand
            return 1.0;
    }
}

/**
 * @brief Calculates the dimensions of a specific plane based on pixel format.
 *
 * @param pix_fmt The pixel format of the frame.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param plane_width Pointer to store the width of the plane.
 * @param plane_height Pointer to store the height of the plane.
 * @param frame_width The width of the frame.
 * @param frame_height The height of the frame.
 * @return int Returns 1 on success, 0 on failure (e.g., unsupported plane).
 */
static int get_plane_dimensions(enum AVPixelFormat pix_fmt, int plane, int *plane_width, int *plane_height, int frame_width, int frame_height) {
    switch (pix_fmt) {
        case AV_PIX_FMT_YUV420P:
            if (plane == 0) { // Y plane
                *plane_width = frame_width;
                *plane_height = frame_height;
            } else { // U and V planes
                *plane_width = frame_width / 2;
                *plane_height = frame_height / 2;
            }
            break;
        case AV_PIX_FMT_YUV422P:
            if (plane == 0) { // Y plane
                *plane_width = frame_width;
                *plane_height = frame_height;
            } else { // U and V planes
                *plane_width = frame_width / 2;
                *plane_height = frame_height;
            }
            break;
        case AV_PIX_FMT_YUV444P:
            // All planes have the same dimensions as the frame
            *plane_width = frame_width;
            *plane_height = frame_height;
            break;
        default:
            // Unsupported pixel format
            return 0;
    }
    return 1; // Success
}

/**
 * @brief Prefetches memory to improve cache utilization.
 *
 * @param ptr Pointer to the memory to prefetch.
 * @param size Size of the memory region to prefetch.
 */
static void prefetch_memory(const void *ptr, size_t size __attribute__((unused))) {
    // Use compiler-specific prefetch instructions
    // GCC and Clang support __builtin_prefetch
    __builtin_prefetch(ptr, 0, 3); // Read, high locality
    // Note: The 'size' parameter is not used here, but can be utilized for more advanced prefetching strategies
}

/**
 * @brief Detects if the CPU supports AVX-512 instructions.
 *
 * @return bool Returns true if AVX-512 is supported, false otherwise.
 */
static bool cpu_supports_avx512(void) {
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_max(0, NULL))
            return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        // Check AVX512F bit (bit 16 of EBX)
        return (ebx & (1 << 16)) != 0;
    #else
        return false;
    #endif
}

/**
 * @brief Detects if the CPU supports NEON instructions.
 *
 * @return bool Returns true if NEON is supported, false otherwise.
 */
static bool cpu_supports_neon(void) {
    #if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        return true;
    #else
        return false;
    #endif
}

/**
 * @brief Checks if the pixel format is supported for SAD calculation.
 *
 * @param pix_fmt The pixel format to check.
 * @return true If supported.
 * @return false Otherwise.
 */
static bool is_supported_pix_fmt(enum AVPixelFormat pix_fmt) {
    // List of supported pixel formats
    switch (pix_fmt) {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV444P:
            return true;
        // Add more supported formats as needed
        default:
            return false;
    }
}

/* ----------------------------
 * Function Implementations
 * ----------------------------
 */

/* Scalar Implementations */

/**
 * @brief Scalar implementation of block extraction.
 *
 * Extracts corresponding blocks from two frames without any SIMD optimizations.
 *
 * @param frame_a Pointer to the first AVFrame (reference frame).
 * @param frame_b Pointer to the second AVFrame (current frame).
 * @param block_a Pointer to the sad_block_size x sad_block_size block buffer from frame A.
 * @param block_b Pointer to the sad_block_size x sad_block_size block buffer from frame B.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_sad_block_scalar(const AVFrame *frame_a, const AVFrame *frame_b,
                                    float *block_a, float *block_b, int x, int y,
                                    int plane, int sad_block_size) {
    if (!frame_a || !frame_b || !block_a || !block_b) return 0;
    
    int width_a, height_a, width_b, height_b;
    if (!get_plane_dimensions(frame_a->format, plane, &width_a, &height_a, frame_a->width, frame_a->height)) {
        return 0;
    }
    if (!get_plane_dimensions(frame_b->format, plane, &width_b, &height_b, frame_b->width, frame_b->height)) {
        return 0;
    }
    
    int stride_a = frame_a->linesize[plane];
    int stride_b = frame_b->linesize[plane];
    uint8_t *plane_data_a = frame_a->data[plane];
    uint8_t *plane_data_b = frame_b->data[plane];
    if (!plane_data_a || !plane_data_b) return 0;
    
    memset(block_a, 0, sizeof(float) * sad_block_size * sad_block_size);
    memset(block_b, 0, sizeof(float) * sad_block_size * sad_block_size);
    
    for (int j = 0; j < sad_block_size; j++) {
        for (int i = 0; i < sad_block_size; i++) {
            int pixel_x = x + i;
            int pixel_y = y + j;
            if (pixel_x < width_a && pixel_y < height_a) {
                block_a[j * sad_block_size + i] = (float)plane_data_a[pixel_y * stride_a + pixel_x];
            }
            if (pixel_x < width_b && pixel_y < height_b) {
                block_b[j * sad_block_size + i] = (float)plane_data_b[pixel_y * stride_b + pixel_x];
            }
        }
    }
    
    return 1;
}

/**
 * @brief Scalar implementation of SAD calculation.
 *
 * Calculates the Sum of Absolute Differences between two blocks.
 *
 * @param block_a Pointer to the first block data.
 * @param block_b Pointer to the second block data.
 * @param sad Pointer to store the calculated SAD.
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int calculate_sad_scalar(const float *block_a, const float *block_b, float *sad, int sad_block_size) {
    if (!block_a || !block_b || !sad) return 0;
    
    float sum = 0.0f;
    for (int i = 0; i < sad_block_size * sad_block_size; i++) {
        sum += fabsf(block_a[i] - block_b[i]);
    }
    
    *sad = sum;
    return 1;
}

/* NEON Implementations */

#if defined(__arm__) || defined(__aarch64__)

/**
 * @brief NEON implementation of block extraction.
 *
 * Extracts corresponding blocks from two frames using NEON intrinsics.
 *
 * @param frame_a Pointer to the first AVFrame (reference frame).
 * @param frame_b Pointer to the second AVFrame (current frame).
 * @param block_a Pointer to the sad_block_size x sad_block_size block buffer from frame A.
 * @param block_b Pointer to the sad_block_size x sad_block_size block buffer from frame B.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_sad_blocks_from_frame_neon(const AVFrame *frame_a, const AVFrame *frame_b,
                                             float *block_a, float *block_b,
                                             int x, int y, int plane, int sad_block_size) {
    if (!frame_a || !frame_b || !block_a || !block_b) return 0;
    
    int width_a, height_a, width_b, height_b;
    if (!get_plane_dimensions(frame_a->format, plane, &width_a, &height_a, frame_a->width, frame_a->height)) {
        return 0;
    }
    if (!get_plane_dimensions(frame_b->format, plane, &width_b, &height_b, frame_b->width, frame_b->height)) {
        return 0;
    }
    
    int stride_a = frame_a->linesize[plane];
    int stride_b = frame_b->linesize[plane];
    uint8_t *plane_data_a = frame_a->data[plane];
    uint8_t *plane_data_b = frame_b->data[plane];
    if (!plane_data_a || !plane_data_b) return 0;
    
    memset(block_a, 0, sizeof(float) * sad_block_size * sad_block_size);
    memset(block_b, 0, sizeof(float) * sad_block_size * sad_block_size);
    
    for (int j = 0; j < sad_block_size; j++) {
        for (int i = 0; i < sad_block_size; i += 4) { // Process 4 pixels at a time
            int pixel_x = x + i;
            int pixel_y = y + j;
            
            // Handle Frame A
            if (pixel_x + 3 < width_a && pixel_y < height_a) {
                uint8x8_t pixels_a = vld1_u8(&plane_data_a[pixel_y * stride_a + pixel_x]);
                float32x4_t pixels_a_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(pixels_a))));
                vst1q_f32(&block_a[j * sad_block_size + i], pixels_a_f32);
            }
            
            // Handle Frame B
            if (pixel_x + 3 < width_b && pixel_y < height_b) {
                uint8x8_t pixels_b = vld1_u8(&plane_data_b[pixel_y * stride_b + pixel_x]);
                float32x4_t pixels_b_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(pixels_b))));
                vst1q_f32(&block_b[j * sad_block_size + i], pixels_b_f32);
            }
        }
    }
    
    return 1;
}

/**
 * @brief NEON implementation of SAD calculation.
 *
 * Calculates the Sum of Absolute Differences between two blocks using NEON intrinsics.
 *
 * @param block_a Pointer to the first block data.
 * @param block_b Pointer to the second block data.
 * @param sad Pointer to store the calculated SAD.
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int calculate_sad_neon(const float *block_a, const float *block_b, float *sad, int sad_block_size) {
    if (!block_a || !block_b || !sad) return 0;
    
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    int total_pixels = sad_block_size * sad_block_size;
    int i;
    for (i = 0; i <= total_pixels - 4; i += 4) {
        float32x4_t a = vld1q_f32(&block_a[i]);
        float32x4_t b = vld1q_f32(&block_b[i]);
        float32x4_t diff = vabdq_f32(a, b); // Absolute difference
        sum_vec = vaddq_f32(sum_vec, diff);  // Accumulate
    }
    
    // Horizontal add
    float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
    
    // Handle remaining pixels
    for (; i < total_pixels; i++) {
        sum += fabsf(block_a[i] - block_b[i]);
    }
    
    *sad = sum;
    return 1; // Success
}

#endif // __arm__ || __aarch64__

/* AVX-512 Implementations */

#if defined(__AVX512F__)

/**
 * @brief AVX-512 implementation of block extraction.
 *
 * Extracts corresponding blocks from two frames using AVX-512 intrinsics.
 *
 * @param frame_a Pointer to the first AVFrame (reference frame).
 * @param frame_b Pointer to the second AVFrame (current frame).
 * @param block_a Pointer to the sad_block_size x sad_block_size block buffer from frame A.
 * @param block_b Pointer to the sad_block_size x sad_block_size block buffer from frame B.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_sad_block_avx512(const AVFrame *frame_a, const AVFrame *frame_b,
                                    float *block_a, float *block_b, int x, int y,
                                    int plane, int sad_block_size) {
    if (!frame_a || !frame_b || !block_a || !block_b) return 0;
    
    int width_a, height_a, width_b, height_b;
    if (!get_plane_dimensions(frame_a->format, plane, &width_a, &height_a, frame_a->width, frame_a->height)) {
        return 0;
    }
    if (!get_plane_dimensions(frame_b->format, plane, &width_b, &height_b, frame_b->width, frame_b->height)) {
        return 0;
    }
    
    int stride_a = frame_a->linesize[plane];
    int stride_b = frame_b->linesize[plane];
    uint8_t *plane_data_a = frame_a->data[plane];
    uint8_t *plane_data_b = frame_b->data[plane];
    if (!plane_data_a || !plane_data_b) return 0;
    
    memset(block_a, 0, sizeof(float) * sad_block_size * sad_block_size);
    memset(block_b, 0, sizeof(float) * sad_block_size * sad_block_size);
    
    for (int j = 0; j < sad_block_size; j++) {
        for (int i = 0; i < sad_block_size; i += 16) { // Process 16 pixels at a time
            int pixel_x = x + i;
            int pixel_y = y + j;
            
            // Handle Frame A
            if (pixel_x + 15 < width_a && pixel_y < height_a) {
                __m512i pixels_a = _mm512_loadu_si512(&plane_data_a[pixel_y * stride_a + pixel_x]);
                __m512 pixels_a_f32 = _mm512_cvtepu8_ps(pixels_a);
                _mm512_storeu_ps(&block_a[j * sad_block_size + i], pixels_a_f32);
            }
            
            // Handle Frame B
            if (pixel_x + 15 < width_b && pixel_y < height_b) {
                __m512i pixels_b = _mm512_loadu_si512(&plane_data_b[pixel_y * stride_b + pixel_x]);
                __m512 pixels_b_f32 = _mm512_cvtepu8_ps(pixels_b);
                _mm512_storeu_ps(&block_b[j * sad_block_size + i], pixels_b_f32);
            }
        }
    }
    
    return 1;
}

/**
 * @brief AVX-512 implementation of SAD calculation.
 *
 * Calculates the Sum of Absolute Differences between two blocks using AVX-512 intrinsics.
 *
 * @param block_a Pointer to the first block data.
 * @param block_b Pointer to the second block data.
 * @param sad Pointer to store the calculated SAD.
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int calculate_sad_avx512(const float *block_a, const float *block_b, float *sad, int sad_block_size) {
    if (!block_a || !block_b || !sad) return 0;
    
    __m512 sum_vec = _mm512_setzero_ps();
    
    int total_pixels = sad_block_size * sad_block_size;
    int i;
    for (i = 0; i <= total_pixels - 16; i += 16) {
        __m512 vec_a = _mm512_loadu_ps(&block_a[i]); // Load 16 floats from block A
        __m512 vec_b = _mm512_loadu_ps(&block_b[i]); // Load 16 floats from block B
        __m512 abs_diff = _mm512_abs_ps(_mm512_sub_ps(vec_a, vec_b)); // Compute absolute differences
        sum_vec = _mm512_add_ps(sum_vec, abs_diff); // Accumulate
    }
    
    // Horizontal add
    float total = _mm512_reduce_add_ps(sum_vec);
    
    // Handle remaining pixels
    for (; i < total_pixels; i++) {
        total += fabsf(block_a[i] - block_b[i]);
    }
    
    *sad = total;
    return 1; // Success
}

#endif // __AVX512F__

/* ----------------------------
 * Initialization Function
 * ----------------------------
 */

/**
 * @brief Initializes the SAD_Functions struct with appropriate function pointers based on CPU capabilities.
 *
 * @param funcs Pointer to the SAD_Functions struct to initialize.
 */
static void initialize_sad_functions(SAD_Functions *funcs) {
    if (cpu_supports_avx512()) {
        #if defined(__AVX512F__)
            funcs->extract_block = extract_sad_block_avx512;
            funcs->calculate_sad = calculate_sad_avx512;
        #else
            // Fallback to scalar if AVX-512 functions are not fully implemented
            funcs->extract_block = extract_sad_block_scalar;
            funcs->calculate_sad = calculate_sad_scalar;
        #endif
    }
    else if (cpu_supports_neon()) {
        #if defined(__arm__) || defined(__aarch64__)
            funcs->extract_block = extract_sad_blocks_from_frame_neon;
            funcs->calculate_sad = calculate_sad_neon;
        #else
            // Fallback to scalar if NEON is not fully implemented
            funcs->extract_block = extract_sad_block_scalar;
            funcs->calculate_sad = calculate_sad_scalar;
        #endif
    }
    else {
        funcs->extract_block = extract_sad_block_scalar;
        funcs->calculate_sad = calculate_sad_scalar;
    }
}

/* ----------------------------
 * SAD Calculation Function
 * ----------------------------
 */

/**
 * @brief Calculates the Sum of Absolute Differences (SAD) between two AVFrames.
 *
 * This function processes each sad_block_size x sad_block_size block of each plane in the frames,
 * extracts corresponding blocks, calculates the SAD, and accumulates the weighted SAD.
 * The final SAD is normalized by the total frame area (width × height).
 *
 * It ensures thread safety by pre-allocating separate block buffers for each thread.
 * Implements an early exit mechanism using OpenMP's cancel clauses in case of critical errors.
 * Utilizes function pointers within a struct to select optimized implementations based on CPU capabilities.
 *
 * @param frame_a Pointer to the first AVFrame (reference frame).
 * @param frame_b Pointer to the second AVFrame (current frame).
 * @param sad_block_size The size of the SAD block (e.g., 8 for 8x8 SAD).
 * @param out_sad Pointer to store the calculated normalized SAD.
 * @return int Returns 0 on success, non-zero error codes on failure.
 */
int calculate_frame_sad(const AVFrame *frame_a, const AVFrame *frame_b, int sad_block_size, float *out_sad) {
    // Validate the input AVFrame pointers
    if (!frame_a) {
        log_error(ERR_NULL_FRAME_A);
        return ERR_NULL_FRAME_A;
    }
    if (!frame_b) {
        log_error(ERR_NULL_FRAME_B);
        return ERR_NULL_FRAME_B;
    }
    
    // Validate the output SAD pointer
    if (!out_sad) {
        log_error(ERR_NULL_OUTPUT);
        return ERR_NULL_OUTPUT;
    }
    
    // Check if the frames' pixel formats are supported
    if (!is_supported_pix_fmt(frame_a->format) || !is_supported_pix_fmt(frame_b->format)) {
        log_error(ERR_UNSUPPORTED_PIX_FMT);
        return ERR_UNSUPPORTED_PIX_FMT;
    }
    
    enum AVPixelFormat pix_fmt_a = frame_a->format;
    enum AVPixelFormat pix_fmt_b = frame_b->format;
    
    // For simplicity, ensure both frames have the same pixel format
    if (pix_fmt_a != pix_fmt_b) {
        log_error(ERR_INVALID_PIX_FMT);
        return ERR_INVALID_PIX_FMT;
    }
    
    int width_a = frame_a->width;
    int height_a = frame_a->height;
    int width_b = frame_b->width;
    int height_b = frame_b->height;
    
    // Ensure both frames have the same dimensions
    if (width_a != width_b || height_a != height_b) {
        log_error(ERR_INVALID_PIX_FMT);
        return ERR_INVALID_PIX_FMT;
    }
    
    enum AVPixelFormat pix_fmt = pix_fmt_a; // Since both are same
    int width = width_a;
    int height = height_a;
    
    // Determine the number of planes based on the pixel format
    int num_planes = av_pix_fmt_count_planes(pix_fmt);
    
    float total_sad = 0.0f; // Variable to accumulate the total SAD
    
    // Determine the number of threads to pre-allocate buffers
    int num_threads = omp_get_max_threads();
    
    // Allocate arrays to hold pointers to blocks for each thread
    float **blocks_a = malloc(num_threads * sizeof(float *));
    float **blocks_b = malloc(num_threads * sizeof(float *));
    
    // Check for successful allocation of buffer arrays
    if (!blocks_a || !blocks_b) {
        log_error(ERR_MEMORY_ALLOCATION);
        free(blocks_a);
        free(blocks_b);
        return ERR_MEMORY_ALLOCATION;
    }
    
    // Allocate aligned memory for each thread's block buffers
    for (int i = 0; i < num_threads; i++) {
        blocks_a[i] = ALIGN_ALLOC(CACHE_LINE_SIZE, sad_block_size * sad_block_size * sizeof(float));
        blocks_b[i] = ALIGN_ALLOC(CACHE_LINE_SIZE, sad_block_size * sad_block_size * sizeof(float));
        
        // Check for successful allocation
        if (!blocks_a[i] || !blocks_b[i]) {
            fprintf(stderr, "Error: Failed to allocate memory for thread %d buffers.\n", i);
            // Free previously allocated buffers
            for (int j = 0; j < i; j++) {
                ALIGN_FREE(blocks_a[j]);
                ALIGN_FREE(blocks_b[j]);
            }
            free(blocks_a);
            free(blocks_b);
            log_error(ERR_MEMORY_ALLOCATION);
            return ERR_MEMORY_ALLOCATION;
        }
    }
    
    // Initialize the function pointers based on CPU capabilities
    SAD_Functions sad_funcs;
    initialize_sad_functions(&sad_funcs);
    
    // Shared flag to indicate if a critical error has occurred
    int error_flag = 0;
    
    // Parallel region begins here
    #pragma omp parallel shared(total_sad, error_flag, sad_funcs, frame_a, frame_b, sad_block_size, num_planes, width, height)
    {
        // Get the thread ID
        int thread_id = omp_get_thread_num();
        
        // Assign each thread its own pre-allocated block buffers
        float *block_a = blocks_a[thread_id];
        float *block_b = blocks_b[thread_id];
        
        float local_sad = 0.0f; // Local SAD accumulator for each thread
        
        // Parallelize the processing of planes and blocks using OpenMP
        // 'collapse(2)' merges the two loops (plane and y) into a single iteration space
        // 'schedule(guided, 16)' dynamically assigns iterations to threads in chunks, optimizing load balancing
        #pragma omp for collapse(2) schedule(guided, 16) nowait
        for (int plane = 0; plane < num_planes; plane++) {
            // Early exit if a critical error has been flagged
            if (__builtin_expect(error_flag, 0)) {
                #pragma omp cancel for
            }
            
            int plane_width, plane_height;
            // Retrieve dimensions for the current plane
            if (!get_plane_dimensions(pix_fmt, plane, &plane_width, &plane_height, width, height)) {
                fprintf(stderr, "Warning: Unable to get dimensions for plane %d.\n", plane);
                continue; // Skip processing this plane if dimensions are invalid
            }
            
            double weight = get_plane_weight(pix_fmt, plane); // Get the weight for the current plane
            
            // Iterate over the plane's blocks in steps of sad_block_size
            for (int y = 0; y < plane_height; y += sad_block_size) {
                for (int x = 0; x < plane_width; x += sad_block_size) {
                    // Early exit if a critical error has been flagged
                    if (__builtin_expect(error_flag, 0)) {
                        #pragma omp cancel for
                    }
                    
                    // Prefetch the block data from both frames to improve cache utilization
                    prefetch_memory(&frame_a->data[plane][y * frame_a->linesize[plane] + x], sad_block_size * sizeof(uint8_t));
                    prefetch_memory(&frame_b->data[plane][y * frame_b->linesize[plane] + x], sad_block_size * sizeof(uint8_t));
                    
                    // Extract corresponding blocks from both frames
                    if (!sad_funcs.extract_block(frame_a, frame_b, block_a, block_b, x, y, plane, sad_block_size)) {
                        fprintf(stderr, "Error: Failed to extract blocks at plane %d, x=%d, y=%d.\n", plane, x, y);
                        // Set the error flag atomically and cancel the loop
                        #pragma omp atomic write
                        error_flag = 1;
                        #pragma omp cancel for;
                    }
                    
                    // Calculate SAD between the extracted blocks
                    float sad = 0.0f;
                    if (!sad_funcs.calculate_sad(block_a, block_b, &sad, sad_block_size)) {
                        fprintf(stderr, "Error: SAD calculation failed at plane %d, x=%d, y=%d.\n", plane, x, y);
                        // Set the error flag atomically and cancel the loop
                        #pragma omp atomic write
                        error_flag = 1;
                        #pragma omp cancel for;
                    }
                    
                    // Accumulate the weighted SAD for this block
                    local_sad += (float)(weight * sad);
                }
            }
        }
        
        // Define cancellation points where threads check if cancellation has been requested
        #pragma omp cancellation point for
        
        // If a critical error has been flagged, skip accumulating SAD
        if (!__builtin_expect(error_flag, 0)) {
            // Atomically add the thread's local SAD to the total SAD
            #pragma omp atomic
            total_sad += local_sad;
        }
    } // End of parallel region
    
    // Free allocated buffers for each thread
    for (int i = 0; i < num_threads; i++) {
        ALIGN_FREE(blocks_a[i]);
        ALIGN_FREE(blocks_b[i]);
    }
    free(blocks_a);
    free(blocks_b);
    
    // If a critical error was detected during parallel processing, return an error code
    if (__builtin_expect(error_flag, 0)) {
        log_error(ERR_PARALLEL_CANCELLATION);
        *out_sad = 0.0f;
        return ERR_PARALLEL_CANCELLATION;
    }
    
    // Calculate the total frame area (width * height) for normalization
    double total_pixels = (double)(width * height);
    
    // Check for zero frame area to avoid division by zero
    if (total_pixels == 0.0) {
        log_error(ERR_ZERO_AREA);
        *out_sad = 0.0f;
        return ERR_ZERO_AREA; // Return specific error code for zero area
    }
    
    // Normalize the total SAD by the frame area to obtain average SAD per pixel
    *out_sad = (float)(total_sad / total_pixels);
    
    return 0; // Indicate successful computation
}

#if defined(__AVX512F__)

/**
 * @brief AVX-512 implementation of SAD calculation.
 *
 * Calculates the Sum of Absolute Differences between two blocks using AVX-512 intrinsics.
 *
 * @param block_a Pointer to the first block data.
 * @param block_b Pointer to the second block data.
 * @param sad Pointer to store the calculated SAD.
 * @param sad_block_size The size of the SAD block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int calculate_sad_avx512(const float *block_a, const float *block_b, float *sad, int sad_block_size) {
    if (!block_a || !block_b || !sad) return 0;
    
    __m512 sum_vec = _mm512_setzero_ps();
    
    int total_pixels = sad_block_size * sad_block_size;
    int i;
    for (i = 0; i <= total_pixels - 16; i += 16) {
        __m512 vec_a = _mm512_loadu_ps(&block_a[i]); // Load 16 floats from block A
        __m512 vec_b = _mm512_loadu_ps(&block_b[i]); // Load 16 floats from block B
        __m512 abs_diff = _mm512_abs_ps(_mm512_sub_ps(vec_a, vec_b)); // Compute absolute differences
        sum_vec = _mm512_add_ps(sum_vec, abs_diff); // Accumulate
    }
    
    // Horizontal add
    float total = _mm512_reduce_add_ps(sum_vec);
    
    // Handle remaining pixels
    for (; i < total_pixels; i++) {
        total += fabsf(block_a[i] - block_b[i]);
    }
    
    *sad = total;
    return 1; // Success
}

#endif // __AVX512F__
