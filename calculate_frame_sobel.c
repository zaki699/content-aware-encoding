/**
 * @file calculate_frame_sobel_energy.c
 * @brief Calculates the Sobel energy of an AVFrame with support for AVX-512, NEON, and Scalar processing.
 *
 * This implementation dynamically selects the best available optimization (AVX-512, NEON, or Scalar)
 * based on the CPU's capabilities at runtime. It ensures optimal performance across diverse hardware
 * platforms while maintaining robustness through comprehensive error handling.
 *
 * Author: Zaki Ahmed
 * Date: 2024-06-20
 */

#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <omp.h>          // For OpenMP parallelization
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>    // For AVX-512 intrinsics
#include <cpuid.h>        // For __get_cpuid_count
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>     // For NEON intrinsics
#endif
#include <stdlib.h>       // For aligned_alloc, free
#include <stdio.h>        // For fprintf, stderr, printf
#include <string.h>       // For memset
#include <stdint.h>       // For uint32_t
#include <stdbool.h>      // For bool type
#include <math.h>         // For fabsf

/* ----------------------------
 * Error Codes
 * ----------------------------
 *
 * Define all error codes as static const integers to ensure they have internal linkage
 * and are confined to this source file.
 */
static const int ERR_NULL_FRAME            = -1;
static const int ERR_NULL_OUTPUT           = -2;
static const int ERR_UNSUPPORTED_PIX_FMT   = -3;
static const int ERR_INVALID_PIX_FMT       = -4;
static const int ERR_MEMORY_ALLOCATION     = -5;
static const int ERR_PARALLEL_CANCELLATION = -6;
static const int ERR_ZERO_AREA             = -7;
static const int ERR_INVALID_PLANE         = -8;
// Add more error codes as needed

/* ----------------------------
 * Structure Definitions
 * ----------------------------
 *
 * Define the Sobel_Functions structure only once to avoid redefinition conflicts.
 */
typedef struct Sobel_Functions {
    /**
     * @brief Function pointer for block extraction.
     *
     * Extracts a block from the AVFrame.
     *
     * @param frame Pointer to the AVFrame.
     * @param block Pointer to the buffer where the block will be stored.
     * @param x The x-coordinate of the top-left corner of the block.
     * @param y The y-coordinate of the top-left corner of the block.
     * @param plane The plane index (0 for Y).
     * @param block_size The size of the Sobel block (should be 3).
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*extract_block)(const AVFrame *frame, float *block, int x, int y, int plane, int block_size);
    
    /**
     * @brief Function pointer for applying the Sobel operator.
     *
     * Applies the Sobel operator to a block and computes the Sobel energy.
     *
     * @param block Pointer to the input block data.
     * @param sobel_output Pointer to store the calculated Sobel energy.
     * @param width Width of the block (should be 3).
     * @param height Height of the block (should be 3).
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*apply_sobel)(const float *block, float *sobel_output, int width, int height);
} Sobel_Functions;

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

// Wrapper function for aligned memory allocation using posix_memalign
void* ALIGN_ALLOC(size_t alignment, size_t size) {
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        return NULL; // Allocation failed
    }
    return ptr;
}

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
static int extract_sobel_block_scalar(const AVFrame *frame, float *block, int x, int y, int plane, int block_size);
static int apply_sobel_scalar(const float *block, float *sobel_output, int width, int height);

/* NEON Implementations */
#if defined(__arm__) || defined(__aarch64__)
static int extract_sobel_block_neon(const AVFrame *frame, float *block, int x, int y, int plane, int block_size);
static int apply_sobel_neon(const float *block, float *sobel_output, int width, int height);
#endif

/* AVX-512 Implementations */
#if defined(__AVX512F__)
static int extract_sobel_block_avx512(const AVFrame *frame, float *block, int x, int y, int plane, int block_size);
static int apply_sobel_avx512(const float *block, float *sobel_output, int width, int height);
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
        case ERR_NULL_FRAME:
            fprintf(stderr, "Error: NULL frame pointer.\n");
            break;
        case ERR_NULL_OUTPUT:
            fprintf(stderr, "Error: NULL output pointer.\n");
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
            fprintf(stderr, "Error: Frame area is zero, cannot normalize Sobel energy.\n");
            break;
        case ERR_INVALID_PLANE:
            fprintf(stderr, "Error: Invalid plane index.\n");
            break;
        default:
            fprintf(stderr, "Error: Unknown error code %d.\n", error_code);
            break;
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
 * @brief Detects if the CPU supports AVX-512 instructions.
 *
 * @return bool Returns true if AVX-512 is supported, false otherwise.
 */
#if defined(__AVX512F__)
static bool cpu_supports_avx512(void) {
    #if defined(__x86_64__)
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
#endif

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
 * @brief Checks if the pixel format is supported for Sobel energy calculation.
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
        case AV_PIX_FMT_YUV444P:
            return true;
        // Add more supported formats as needed
        default:
            return false;
    }
}

/* ----------------------------
 * Initialization Function
 * ----------------------------
 */

/**
 * @brief Initializes the Sobel_Functions struct with appropriate function pointers based on CPU capabilities.
 *
 * @param funcs Pointer to the Sobel_Functions struct to initialize.
 */
static void initialize_sobel_functions(Sobel_Functions *funcs) {
    #if defined(__AVX512F__)
        if (cpu_supports_avx512()) {
            funcs->extract_block = extract_sobel_block_avx512;
            funcs->apply_sobel = apply_sobel_avx512;
            return;
        }
    #endif

    #if defined(__arm__) || defined(__aarch64__)
        if (cpu_supports_neon()) {
            funcs->extract_block = extract_sobel_block_neon;
            funcs->apply_sobel = apply_sobel_neon;
            return;
        }
    #endif

    // Fallback to scalar implementations
    funcs->extract_block = extract_sobel_block_scalar;
    funcs->apply_sobel = apply_sobel_scalar;
}

/* ----------------------------
 * Scalar Implementations
 * ---------------------------- */

/**
 * @brief Scalar implementation of block extraction.
 *
 * Extracts a 3x3 block from the specified plane at (x, y).
 *
 * @param frame Pointer to the AVFrame.
 * @param block Pointer to the buffer where the block will be stored.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y).
 * @param block_size The size of the Sobel block (should be 3).
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_sobel_block_scalar(const AVFrame *frame, float *block, int x, int y, int plane, int block_size) {
    if (!frame || !block) return 0;
    
    if (block_size != 3) return 0; // Currently supporting only 3x3 blocks

    int width, height;
    if (!get_plane_dimensions(frame->format, plane, &width, &height, frame->width, frame->height)) {
        return 0;
    }
    
    int stride = frame->linesize[plane];
    uint8_t *plane_data = frame->data[plane];
    if (!plane_data) return 0;
    
    memset(block, 0, sizeof(float) * block_size * block_size);
    
    for (int j = 0; j < block_size; j++) {
        for (int i = 0; i < block_size; i++) {
            int pixel_x = x + i;
            int pixel_y = y + j;
            if (pixel_x < width && pixel_y < height) {
                block[j * block_size + i] = (float)plane_data[pixel_y * stride + pixel_x];
            }
        }
    }
    
    return 1;
}

/**
 * @brief Scalar implementation of Sobel application.
 *
 * Computes the Sobel gradient magnitude for a 3x3 block.
 *
 * @param block Pointer to the input block data.
 * @param sobel_output Pointer to store the calculated Sobel energy.
 * @param width Width of the block (should be 3).
 * @param height Height of the block (should be 3).
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_sobel_scalar(const float *block, float *sobel_output, int width, int height) {
    if (!block || !sobel_output) return 0;
    
    if (width != 3 || height != 3) return 0; // Currently supporting only 3x3 blocks

    float total_sobel = 0.0f;

    // Apply Sobel operator to the center pixel (y=1, x=1)
    float gx = (-1.0f * block[0]) + (0.0f * block[1]) + (+1.0f * block[2]) +
                (-2.0f * block[3]) + (0.0f * block[4]) + (+2.0f * block[5]) +
                (-1.0f * block[6]) + (0.0f * block[7]) + (+1.0f * block[8]);

    float gy = (+1.0f * block[0]) + (+2.0f * block[1]) + (+1.0f * block[2]) +
                (0.0f * block[3]) + (0.0f * block[4]) + (0.0f * block[5]) +
                (-1.0f * block[6]) + (-2.0f * block[7]) + (-1.0f * block[8]);

    // Compute gradient magnitude using |Gx| + |Gy| approximation
    float magnitude = fabsf(gx) + fabsf(gy);
    total_sobel += magnitude;

    *sobel_output = total_sobel;
    return 1;
}

/* ----------------------------
 * NEON Implementations
 * ----------------------------
 */
#if defined(__arm__) || defined(__aarch64__)

/**
 * @brief NEON implementation of block extraction.
 *
 * Extracts a 3x3 block from the specified plane at (x, y) using NEON intrinsics.
 *
 * @param frame Pointer to the AVFrame.
 * @param block Pointer to the buffer where the block will be stored.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y).
 * @param block_size The size of the Sobel block (should be 3).
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_sobel_block_neon(const AVFrame *frame, float *block, int x, int y, int plane, int block_size) {
    if (block_size != 3) {
        return extract_sobel_block_scalar(frame, block, x, y, plane, block_size);
    }

    int width, height;
    if (!get_plane_dimensions(frame->format, plane, &width, &height, frame->width, frame->height)) {
        return 0;
    }

    int stride = frame->linesize[plane];
    uint8_t *plane_data = frame->data[plane];
    if (!plane_data) return 0;

    memset(block, 0, sizeof(float) * block_size * block_size);

    for (int j = 0; j < block_size; j++) {
        int pixel_y = y + j;
        if (pixel_y >= height) continue;

        int pixel_x = x;
        if (pixel_x >= width) continue;

        // Load 3 pixels into a NEON register
        uint8x8_t src_u8 = vld1_u8(&plane_data[pixel_y * stride + pixel_x]);

        // Convert to float32
        float32x4_t pixels_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_u8))));

        // Store the first three floats
        vst1q_lane_f32(&block[j * block_size + 0], pixels_f32, 0);
        vst1q_lane_f32(&block[j * block_size + 1], pixels_f32, 1);
        vst1q_lane_f32(&block[j * block_size + 2], pixels_f32, 2);
    }

    return 1;
}

/**
 * @brief NEON implementation of Sobel application.
 *
 * Computes the Sobel gradient magnitude for a 3x3 block using NEON intrinsics.
 *
 * @param block Pointer to the input block data.
 * @param sobel_output Pointer to store the calculated Sobel energy.
 * @param width Width of the block (should be 3).
 * @param height Height of the block (should be 3).
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_sobel_neon(const float *block, float *sobel_output, int width, int height) {
    if (!block || !sobel_output) return 0;
    if (width != 3 || height != 3) return 0;

    float total_sobel = 0.0f;

    // Load the 3x3 block into NEON registers
    float32x4_t row0 = vld1q_f32(&block[0]); // block[0], block[1], block[2], unused
    float32x4_t row1 = vld1q_f32(&block[3]); // block[3], block[4], block[5], unused
    float32x4_t row2 = vld1q_f32(&block[6]); // block[6], block[7], block[8], unused

    // Extract individual elements
    float p0 = vgetq_lane_f32(row0, 0);
    float p1 = vgetq_lane_f32(row0, 1);
    float p2 = vgetq_lane_f32(row0, 2);
    float p3 = vgetq_lane_f32(row1, 0);
    float p4 = vgetq_lane_f32(row1, 1);
    float p5 = vgetq_lane_f32(row1, 2);
    float p6 = vgetq_lane_f32(row2, 0);
    float p7 = vgetq_lane_f32(row2, 1);
    float p8 = vgetq_lane_f32(row2, 2);

    // Apply Sobel Gx
    float gx = (-1.0f * p0) + (0.0f * p1) + (+1.0f * p2) +
                (-2.0f * p3) + (0.0f * p4) + (+2.0f * p5) +
                (-1.0f * p6) + (0.0f * p7) + (+1.0f * p8);

    // Apply Sobel Gy
    float gy = (+1.0f * p0) + (+2.0f * p1) + (+1.0f * p2) +
                (0.0f * p3) + (0.0f * p4) + (0.0f * p5) +
                (-1.0f * p6) + (-2.0f * p7) + (-1.0f * p8);

    // Compute gradient magnitude using |Gx| + |Gy| approximation
    float magnitude = fabsf(gx) + fabsf(gy);
    total_sobel += magnitude;

    *sobel_output = total_sobel;
    return 1;
}

#endif // __arm__ || __aarch64__

/* ----------------------------
 * AVX-512 Implementations
 * ----------------------------
 */
#if defined(__AVX512F__)

/**
 * @brief AVX-512 implementation of block extraction.
 *
 * Extracts a 3x3 block from the specified plane at (x, y) using AVX-512 intrinsics.
 *
 * @param frame Pointer to the AVFrame.
 * @param block Pointer to the buffer where the block will be stored.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y).
 * @param block_size The size of the Sobel block (should be 3).
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_sobel_block_avx512(const AVFrame *frame, float *block, int x, int y, int plane, int block_size) {
    if (block_size != 3) {
        return extract_sobel_block_scalar(frame, block, x, y, plane, block_size);
    }

    int width, height;
    if (!get_plane_dimensions(frame->format, plane, &width, &height, frame->width, frame->height)) {
        return 0;
    }

    int stride = frame->linesize[plane];
    uint8_t *plane_data = frame->data[plane];
    if (!plane_data) {
        return 0;
    }

    memset(block, 0, sizeof(float) * block_size * block_size);

    for (int j = 0; j < block_size; j++) {
        int pixel_y = y + j;
        if (pixel_y >= height) continue;

        int pixel_x = x;
        if (pixel_x >= width) continue;

        for (int i = 0; i < block_size; i++) {
            int pixel_x_current = pixel_x + i;
            if (pixel_x_current >= width) continue;

            // Load one pixel using AVX-512 intrinsics
            uint8_t pixel = plane_data[pixel_y * stride + pixel_x_current];
            block[j * block_size + i] = (float)pixel;
        }
    }

    return 1;
}

/**
 * @brief AVX-512 implementation of Sobel application.
 *
 * Computes the Sobel gradient magnitude for a 3x3 block using AVX-512 intrinsics.
 *
 * @param block Pointer to the input block data.
 * @param sobel_output Pointer to store the calculated Sobel energy.
 * @param width Width of the block (should be 3).
 * @param height Height of the block (should be 3).
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_sobel_avx512(const float *block, float *sobel_output, int width, int height) {
    if (!block || !sobel_output) return 0;
    if (width != 3 || height != 3) return 0;

    // Load the 3x3 block into AVX-512 registers
    __m512 row0 = _mm512_setr_ps(block[0], block[1], block[2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    __m512 row1 = _mm512_setr_ps(block[3], block[4], block[5], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    __m512 row2 = _mm512_setr_ps(block[6], block[7], block[8], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // Extract individual elements
    float p0 = _mm512_cvtss_f32(_mm512_castps512_ps128(row0));
    float p1 = _mm512_cvtss_f32(_mm512_extractf32x4_ps(row0, 0));
    float p2 = _mm512_cvtss_f32(_mm512_extractf32x4_ps(row0, 0));
    float p3 = _mm512_cvtss_f32(_mm512_castps512_ps128(row1));
    float p4 = _mm512_cvtss_f32(_mm512_extractf32x4_ps(row1, 0));
    float p5 = _mm512_cvtss_f32(_mm512_extractf32x4_ps(row1, 0));
    float p6 = _mm512_cvtss_f32(_mm512_castps512_ps128(row2));
    float p7 = _mm512_cvtss_f32(_mm512_extractf32x4_ps(row2, 0));
    float p8 = _mm512_cvtss_f32(_mm512_extractf32x4_ps(row2, 0));

    // Apply Sobel Gx
    float gx = (-1.0f * p0) + (0.0f * p1) + (+1.0f * p2) +
                (-2.0f * p3) + (0.0f * p4) + (+2.0f * p5) +
                (-1.0f * p6) + (0.0f * p7) + (+1.0f * p8);

    // Apply Sobel Gy
    float gy = (+1.0f * p0) + (+2.0f * p1) + (+1.0f * p2) +
                (0.0f * p3) + (0.0f * p4) + (0.0f * p5) +
                (-1.0f * p6) + (-2.0f * p7) + (-1.0f * p8);

    // Compute gradient magnitude using |Gx| + |Gy| approximation
    float magnitude = fabsf(gx) + fabsf(gy);

    *sobel_output = magnitude;
    return 1;
}


#endif // __AVX512F__

/* ----------------------------
 * Sobel Energy Calculation Function
 * ----------------------------
 */

/**
 * @brief Calculates the Sobel energy of an AVFrame.
 *
 * This function processes each 3x3 block of the Y plane in the frame,
 * applies the Sobel operator, calculates the energy of the Sobel magnitudes, and accumulates
 * the total Sobel energy.
 *
 * It ensures thread safety by pre-allocating separate block buffers for each thread.
 * Implements an early exit mechanism using OpenMP's cancel clauses in case of critical errors.
 * Utilizes function pointers within a struct to select optimized implementations based on CPU capabilities.
 *
 * @param frame Pointer to the AVFrame containing the image data.
 * @param block_size The size of the Sobel block (should be 3).
 * @param out_energy Pointer to store the calculated normalized energy.
 * @return int Returns 0 on success, non-zero error codes on failure.
 */
int calculate_frame_sobel_energy(const AVFrame *frame, int block_size, float *out_energy) {
    if (!frame || !out_energy) {
        log_error(ERR_NULL_FRAME);
        return ERR_NULL_FRAME;
    }

    if (!is_supported_pix_fmt(frame->format)) {
        log_error(ERR_UNSUPPORTED_PIX_FMT);
        return ERR_UNSUPPORTED_PIX_FMT;
    }

    float total_sobel = 0.0f;
    int num_threads = omp_get_max_threads();

    // Allocate arrays to hold pointers to blocks for each thread
    float **blocks = malloc(num_threads * sizeof(float *));
    if (!blocks) {
        log_error(ERR_MEMORY_ALLOCATION);
        return ERR_MEMORY_ALLOCATION;
    }

    // Allocate buffers for each thread
    for (int i = 0; i < num_threads; i++) {
        blocks[i] = ALIGN_ALLOC(CACHE_LINE_SIZE, block_size * block_size * sizeof(float));
        if (!blocks[i]) {
            fprintf(stderr, "Memory allocation failed for thread %d buffers.\n", i);
            // Free previously allocated blocks to avoid memory leaks
            for (int j = 0; j < i; j++) {
                ALIGN_FREE(blocks[j]);
            }
            free(blocks);
            log_error(ERR_MEMORY_ALLOCATION);
            return ERR_MEMORY_ALLOCATION;
        }
    }

    // Initialize function pointers based on CPU capabilities
    Sobel_Functions sobel_funcs;
    initialize_sobel_functions(&sobel_funcs);

    // Parallel region begins here
    #pragma omp parallel shared(total_sobel, sobel_funcs, frame, block_size, blocks) default(none) reduction(+:total_sobel)
    {
        int thread_id = omp_get_thread_num();
        float *block = blocks[thread_id];

        // Parallelize the processing of blocks using OpenMP
        #pragma omp for collapse(2) schedule(dynamic, 16)
        for (int y = 0; y < frame->height - 2; y++) { // -2 to ensure 3x3 blocks
            for (int x = 0; x < frame->width - 2; x++) { // -2 to ensure 3x3 blocks
                // Extract the block (assuming plane 0 - Y)
                if (!sobel_funcs.extract_block(frame, block, x, y, 0, block_size)) {
                    log_error(ERR_INVALID_PLANE);
                    #pragma omp cancel for
                }

                float sobel_val = 0.0f;
                // Apply the Sobel operator
                if (!sobel_funcs.apply_sobel(block, &sobel_val, block_size, block_size)) {
                    log_error(ERR_PARALLEL_CANCELLATION);
                    #pragma omp cancel for
                }

                // Accumulate the Sobel energy
                total_sobel += sobel_val;
            }
        }

        // Handle cancellation if any
        #pragma omp cancellation point for
    } // End of parallel region

    // Free allocated buffers for each thread
    for (int i = 0; i < num_threads; i++) {
        ALIGN_FREE(blocks[i]);
    }
    free(blocks);

    // Calculate the number of Sobel operations performed
    double num_operations = (double)(frame->width - 2) * (frame->height - 2);

    // Check for zero frame area to avoid division by zero
    if (num_operations <= 0.0) {
        log_error(ERR_ZERO_AREA);
        *out_energy = 0.0f;
        return ERR_ZERO_AREA; // Return specific error code for zero area
    }

    // Normalize the total Sobel energy by the number of Sobel operations to obtain average energy per pixel
    *out_energy = (float)(total_sobel / num_operations);

    return 0; // Indicate successful computation
}
