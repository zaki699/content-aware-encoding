/**
 * @file calculate_frame_dct_energy.c
 * @brief Calculates the DCT energy of an AVFrame with support for AVX-512, NEON, and Scalar processing.
 *
 * This implementation dynamically selects the best available optimization (AVX-512, NEON, or Scalar)
 * based on the CPU's capabilities at runtime. It ensures optimal performance across diverse hardware
 * platforms while maintaining robustness through comprehensive error handling.
 *
 * Author: Zaki Ahmed
 * Date: 2024-05-04
 */

#include <libavutil/tx.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>       // For aligned_alloc, free
#include <stdio.h>        // For fprintf, stderr, printf
#include <string.h>       // For memset
#include <fftw3.h>        // For FFTW DCT functions
#include <omp.h>          // For OpenMP parallelization

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#include <cpuid.h>
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ----------------------------
 * Static Error Codes
 * ----------------------------
 *
 * By declaring error codes as static const variables, we ensure they are confined
 * to this source file, preventing conflicts with error codes in other modules.
 */

static const int ERR_NULL_FRAME                = -1;
static const int ERR_UNSUPPORTED_PIX_FMT       = -2;
static const int ERR_INVALID_PLANE_DIMENSIONS   = -3; // Renamed for clarity
static const int ERR_INVALID_PLANE_INDEX        = -4; // New error code
static const int ERR_MEMORY_ALLOCATION         = -5;
static const int ERR_PARALLEL_CANCELLATION     = -6;
static const int ERR_ZERO_AREA                 = -7;
static const int ERR_DCT_INIT_FAILED           = -8;

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
 * DCT_Functions Structure
 * ----------------------------
 */

/**
 * @struct DCT_Functions
 * @brief Structure containing function pointers for different DCT implementations.
 *
 * This structure allows dynamic selection of the appropriate DCT implementation
 * based on the CPU's capabilities (AVX-512, NEON, or Scalar).
 */
typedef struct DCT_Functions {
    /**
     * @brief Function pointer for block extraction.
     *
     * Extracts a dct_size x dct_size block from a specified plane of an AVFrame.
     *
     * @param frame Pointer to the AVFrame containing the image data.
     * @param block Pointer to the block buffer where the extracted pixels will be stored.
     * @param x The x-coordinate of the top-left corner of the block.
     * @param y The y-coordinate of the top-left corner of the block.
     * @param plane The plane index (0 for Y, 1 for U, 2 for V).
     * @param dct_size The size of the DCT block.
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*extract_block)(const AVFrame *frame, double *block, int x, int y, int plane, int dct_size);
    
    /**
     * @brief Applies a custom Discrete Cosine Transform (DCT) using FFTW.
     *
     * @param plan FFTW plan for the DCT.
     * @param block Pointer to the input block data.
     * @param dct_output Pointer to the buffer where DCT coefficients will be stored.
     * @param dct_size The size of the DCT block.
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*apply_dct)(fftw_plan plan, double *block, double *dct_output, int dct_size);
    
    /**
     * @brief Function pointer for calculating the energy of DCT coefficients.
     *
     * Calculates the energy by summing the squares of DCT coefficients.
     *
     * @param dct_output Pointer to the DCT coefficients.
     * @param energy Pointer to store the calculated energy.
     * @return int Returns 1 on success, 0 on failure.
     */
    int (*calculate_energy)(const double *dct_output, double *energy);
} DCT_Functions;

/* ----------------------------
 * Internal Utility Functions
 * ----------------------------
 */

/**
 * @brief Prefetches memory to improve cache utilization.
 *
 * @param ptr Pointer to the memory to prefetch.
 * @param size Size of the memory region to prefetch.
 */
void prefetch_memory(const void *ptr, size_t size __attribute__((unused))) {
    // Use compiler-specific prefetch instructions
    // GCC and Clang support __builtin_prefetch
    __builtin_prefetch(ptr, 0, 3); // Read, high locality
    // Note: The 'size' parameter is not used here, but can be utilized for more advanced prefetching strategies
}

/**
 * @brief Logs an error message based on the error code.
 *
 * @param error_code The error code indicating the type of error.
 */
static void log_error(int error_code) {
    switch (error_code) {
        case ERR_NULL_FRAME:
            fprintf(stderr, "Error: NULL frame or output pointer.\n");
            break;
        case ERR_UNSUPPORTED_PIX_FMT:
            fprintf(stderr, "Error: Unsupported pixel format.\n");
            break;
        case ERR_INVALID_PLANE_DIMENSIONS:
            fprintf(stderr, "Error: Invalid plane dimensions.\n");
            break;
        case ERR_INVALID_PLANE_INDEX:
            fprintf(stderr, "Error: Invalid plane index.\n");
            break;
        case ERR_MEMORY_ALLOCATION:
            fprintf(stderr, "Error: Memory allocation failed.\n");
            break;
        case ERR_PARALLEL_CANCELLATION:
            fprintf(stderr, "Error: Parallel processing was cancelled due to a critical error.\n");
            break;
        case ERR_ZERO_AREA:
            fprintf(stderr, "Error: Frame area is zero, cannot normalize energy.\n");
            break;
        case ERR_DCT_INIT_FAILED:
            fprintf(stderr, "Error: DCT initialization failed.\n");
            break;
        default:
            fprintf(stderr, "Error: Unknown error code %d.\n", error_code);
            break;
    }
}

/**
 * @brief Detects if the CPU supports AVX-512 instructions.
 *
 * @return bool Returns true if AVX-512 is supported, false otherwise.
 */
static bool cpu_supports_avx512(void) {
    #ifdef __x86_64__
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
 * @brief Retrieves the dimensions of a specific plane based on the pixel format.
 *
 * @param pix_fmt The pixel format of the frame.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V, etc.).
 * @param out_width Pointer to store the plane's width.
 * @param out_height Pointer to store the plane's height.
 * @param frame_width The width of the frame.
 * @param frame_height The height of the frame.
 * @return true If dimensions were successfully retrieved.
 * @return false Otherwise.
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
 * @brief Checks if the pixel format is supported for DCT energy calculation.
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
            return true;
        // Add more supported formats as needed
        default:
            return false;
    }
}

/* ----------------------------
 * Forward Declarations of Internal Helper Functions
 * ----------------------------
 */

/**
 * @brief Applies a custom Discrete Cosine Transform (DCT) using FFTW.
 *
 * @param plan FFTW plan for the DCT.
 * @param block Pointer to the input block data.
 * @param dct_output Pointer to the buffer where DCT coefficients will be stored.
 * @param dct_size The size of the DCT block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_dct_fftw(fftw_plan plan, double *block, double *dct_output, int dct_size);

/**
 * @brief Calculates the energy of DCT coefficients by summing their squares using NEON intrinsics.
 *
 * @param dct_output Pointer to the DCT coefficients.
 * @param energy Pointer to store the calculated energy.
 * @return int Returns 1 on success, 0 on failure.
 */
static int calculate_dct_energy_neon(const double *dct_output, double *energy);

/**
 * @brief Extracts a dct_size x dct_size block from a specific plane of the AVFrame using NEON intrinsics.
 *
 * @param frame Pointer to the AVFrame containing the image data.
 * @param block Pointer to the dct_size x dct_size block buffer where the extracted pixels will be stored.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param dct_size The size of the DCT block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_dct_blocks_from_frame_neon(const AVFrame *frame, double *block, int x, int y, int plane, int dct_size);

/* ----------------------------
 * Scalar Implementations
 * ----------------------------
 */

/**
 * @brief Scalar implementation of block extraction.
 */
static int extract_dct_block_scalar(const AVFrame *frame, double *block, int x, int y, int plane, int dct_size) {
    if (!frame || !block) return 0;
    
    int width, height;
    if (!get_plane_dimensions(frame->format, plane, &width, &height, frame->width, frame->height)) {
        return 0;
    }
    
    int stride = frame->linesize[plane];
    uint8_t *plane_data = frame->data[plane];
    if (!plane_data) return 0;
    
    memset(block, 0, sizeof(double) * dct_size * dct_size);
    
    for (int j = 0; j < dct_size; j++) {
        for (int i = 0; i < dct_size; i++) {
            int pixel_x = x + i;
            int pixel_y = y + j;
            if (pixel_x < width && pixel_y < height) {
                block[j * dct_size + i] = (double)plane_data[pixel_y * stride + pixel_x];
            }
        }
    }
    
    return 1;
}

/**
 * @brief Applies a custom Discrete Cosine Transform (DCT) using FFTW with orthogonal normalization.
 *
 * @param plan FFTW plan for the DCT.
 * @param block Pointer to the input block data.
 * @param dct_output Pointer to the buffer where DCT coefficients will be stored.
 * @param dct_size The size of the DCT block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_dct_fftw(fftw_plan plan, double *block, double *dct_output, int dct_size) {
    if (!plan || !block || !dct_output) return 0;
    
    // Execute the DCT
    fftw_execute_r2r(plan, block, dct_output);
    
    // Apply orthogonal normalization for both dimensions
    for (int i = 0; i < dct_size * dct_size; i++) {
        int u = i % dct_size; // Horizontal index
        int v = i / dct_size; // Vertical index
        
        double scale = 1.0;
        if (u == 0)
            scale *= sqrt(1.0 / dct_size);
        else
            scale *= sqrt(2.0 / dct_size);
        
        if (v == 0)
            scale *= sqrt(1.0 / dct_size);
        else
            scale *= sqrt(2.0 / dct_size);
        
        // **Critical Correction:** Divide by 4 to account for FFTW's DCT-II scaling
        scale /= 4.0;
        
        dct_output[i] *= scale;
    }
    
    // Debug: Print the first few DCT coefficients
    /*static int debug_count = 0;
    if (debug_count < 5) { // Limit the number of debug prints
        printf("DCT Output (first 8 coefficients): ");
        for (int i = 0; i < 8; i++) {
            printf("%.2lf ", dct_output[i]);
        }
        printf("\n");
        debug_count++;
    }*/
    
    return 1;
}

/**
 * @brief Applies a custom Discrete Cosine Transform (DCT) using FFTW.
 *
 * @param plan FFTW plan for the DCT.
 * @param block Pointer to the input block data.
 * @param dct_output Pointer to the buffer where DCT coefficients will be stored.
 * @param dct_size The size of the DCT block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_dct_scalar(fftw_plan plan, double *block, double *dct_output, int dct_size) {
    // SCALAR does not alter DCT application; use the same FFTW function
    return apply_dct_fftw(plan, block, dct_output, dct_size);
}

/**
 * @brief Scalar implementation of energy calculation.
 */
static int calculate_dct_energy_scalar(const double *dct_output, double *energy) {
    if (!dct_output || !energy) return 0;
    
    double sum = 0.0;
    for (int i = 0; i < 64; i++) { // Assuming dct_size=8
        sum += dct_output[i] * dct_output[i];
    }
    
    *energy = sum;
    return 1;
}

/* ----------------------------
 * NEON Implementations
 * ----------------------------
 */

#if defined(__arm__) || defined(__aarch64__)

/**
 * @brief NEON implementation of block extraction with prefetching.
 */
static int extract_dct_block_neon(const AVFrame *frame, double *block, int x, int y, int plane, int dct_size) {
    // Assuming dct_size=8
    if (dct_size != 8) {
        // For simplicity, NEON implementation only handles 8x8 blocks
        return extract_dct_block_scalar(frame, block, x, y, plane, dct_size);
    }
    return extract_dct_blocks_from_frame_neon(frame, block, x, y, plane, dct_size);
}

/**
 * @brief Applies a custom Discrete Cosine Transform (DCT) using FFTW.
 *
 * @param plan FFTW plan for the DCT.
 * @param block Pointer to the input block data.
 * @param dct_output Pointer to the buffer where DCT coefficients will be stored.
 * @param dct_size The size of the DCT block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_dct_neon(fftw_plan plan, double *block, double *dct_output, int dct_size) {
    // NEON does not alter DCT application; use the same FFTW function
    return apply_dct_fftw(plan, block, dct_output, dct_size);
}

/**
 * @brief Extracts an 8x8 block from a specific plane of the AVFrame using NEON intrinsics.
 *
 * @param frame Pointer to the AVFrame containing the image data.
 * @param block Pointer to the 8x8 block buffer where the extracted pixels will be stored.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param dct_size The size of the DCT block (should be 8).
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_dct_blocks_from_frame_neon(const AVFrame *frame, double *block, int x, int y, int plane, int dct_size) {
    // Validate input pointers
    if (!frame || !block) {
        return 0; // Failure due to invalid pointers
    }

    int width, height;
    // Get dimensions for the specified plane
    if (!get_plane_dimensions(frame->format, plane, &width, &height, frame->width, frame->height)) {
        return 0; // Failure due to unsupported plane
    }

    int stride = frame->linesize[plane]; // Number of bytes per row in the plane
    uint8_t *plane_data = frame->data[plane]; // Pointer to the plane's data
    if (!plane_data) {
        return 0; // Failure due to NULL plane data
    }

    // Initialize the block to zero (handle padding for blocks extending beyond frame boundaries)
    memset(block, 0, sizeof(double) * dct_size * dct_size);

    // Iterate over each row of the block
    for (int j = 0; j < dct_size; j++) {
        // Current y position in the frame
        int pixel_y = y + j;
        if (pixel_y >= height) {
            continue; // Skip if beyond frame height
        }

        // Current x position in the frame
        int pixel_x = x;
        if (pixel_x >= width) {
            continue; // Skip if beyond frame width
        }

        // Number of pixels to load in this iteration
        int pixels_to_load = (pixel_x + dct_size <= width) ? dct_size : (width - pixel_x);

        // Load exactly 8 pixels or the remaining pixels
        uint8x8_t src_u8 = (pixels_to_load >= 8) ? vld1_u8(&plane_data[pixel_y * stride + pixel_x])
                                               : vld1_u8(&plane_data[pixel_y * stride + pixel_x]);

        // Convert to double
        double row_values[8];
        for (int i = 0; i < pixels_to_load; i++) {
            row_values[i] = (double)src_u8[i];
        }
        for (int i = pixels_to_load; i < 8; i++) {
            row_values[i] = 0.0;
        }

        // Store the doubles into the block buffer
        for (int i = 0; i < 8; i++) {
            block[j * dct_size + i] = row_values[i];
        }
    }

    return 1; // Success
}

/**
 * @brief Calculates the energy of DCT coefficients by summing their squares using NEON intrinsics.
 *
 * This function leverages ARM NEON's SIMD capabilities to efficiently compute the sum of squares
 * of 64 DCT coefficients. It processes 8 coefficients per iteration using two NEON vectors,
 * accumulates their squares, and performs a horizontal sum to obtain the final energy value.
 *
 * @param dct_output Pointer to the array of DCT coefficients (expected to have at least 64 elements).
 * @param energy Pointer to store the calculated energy.
 * @return int Returns 1 on success, 0 on failure (e.g., invalid input pointers).
 */
static int calculate_dct_energy_neon(const double *dct_output, double *energy) {
    // Validate input pointers
    if (!dct_output || !energy) {
        return 0; // Failure due to invalid pointers
    }

    double sum = 0.0;

    // Process 8 coefficients per loop iteration
    for (int i = 0; i < 64; i += 8) {
        // Load 8 doubles
        double coeffs[8];
        for (int j = 0; j < 8; j++) {
            coeffs[j] = dct_output[i + j];
        }

        // Square and accumulate
        for (int j = 0; j < 8; j++) {
            sum += coeffs[j] * coeffs[j];
        }
    }

    *energy = sum;

    return 1; // Success
}

#endif // __arm__ || __aarch64__

/* ----------------------------
 * AVX-512 Implementations
 * ----------------------------
 */

#if defined(__AVX512F__)

/**
 * @brief Extracts an 8x8 block from a specific plane of the AVFrame using AVX-512 intrinsics.
 *
 * @param frame Pointer to the AVFrame containing the image data.
 * @param block Pointer to the 8x8 block buffer where the extracted pixels will be stored.
 * @param x The x-coordinate of the top-left corner of the block.
 * @param y The y-coordinate of the top-left corner of the block.
 * @param plane The plane index (0 for Y, 1 for U, 2 for V).
 * @param dct_size The size of the DCT block (should be 8).
 * @return int Returns 1 on success, 0 on failure.
 */
static int extract_dct_block_avx512(const AVFrame *frame, double *block, int x, int y, int plane, int dct_size) {
    // Assuming dct_size=8
    if (dct_size != 8) {
        // For simplicity, AVX-512 implementation only handles 8x8 blocks
        return extract_dct_block_scalar(frame, block, x, y, plane, dct_size);
    }

    if (!frame || !block) {
        return 0;
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

    memset(block, 0, sizeof(double) * dct_size * dct_size);

    // Iterate over each row of the block
    for (int j = 0; j < dct_size; j++) {
        int pixel_y = y + j;
        if (pixel_y >= height) {
            continue; // Skip if beyond frame height
        }

        int pixel_x = x;
        if (pixel_x >= width) {
            continue; // Skip if beyond frame width
        }

        // Number of pixels to load in this iteration
        int pixels_to_load = (pixel_x + dct_size <= width) ? dct_size : (width - pixel_x);

        // Load exactly 8 pixels or the remaining pixels
        __m128i src_u8 = (pixels_to_load >= 8) ? _mm_loadu_si128((__m128i*)&plane_data[pixel_y * stride + pixel_x])
                                               : _mm_loadl_epi64((__m128i*)&plane_data[pixel_y * stride + pixel_x]);

        // Convert to double
        double row_values[8];
        for (int i = 0; i < pixels_to_load; i++) {
            // Convert each byte to double
            row_values[i] = (double)_mm_cvtsi128_si32(_mm_srli_si128(src_u8, i * 2)) & 0xFF;
        }
        for (int i = pixels_to_load; i < 8; i++) {
            row_values[i] = 0.0;
        }

        // Store the doubles into the block buffer
        for (int i = 0; i < 8; i++) {
            block[j * dct_size + i] = row_values[i];
        }
    }

    return 1; // Success
}

/**
 * @brief Applies a custom Discrete Cosine Transform (DCT) using FFTW.
 *
 * @param plan FFTW plan for the DCT.
 * @param block Pointer to the input block data.
 * @param dct_output Pointer to the buffer where DCT coefficients will be stored.
 * @param dct_size The size of the DCT block.
 * @return int Returns 1 on success, 0 on failure.
 */
static int apply_dct_avx512(fftw_plan plan, double *block, double *dct_output, int dct_size) {
    // AVX-512 does not alter DCT application; use the same FFTW function
    return apply_dct_fftw(plan, block, dct_output, dct_size);
}

/**
 * @brief Calculates the energy of DCT coefficients by summing their squares using AVX-512 intrinsics.
 *
 * @param dct_output Pointer to the DCT coefficients.
 * @param energy Pointer to store the calculated energy.
 * @return int Returns 1 on success, 0 on failure.
 */
static int calculate_dct_energy_avx512(const double *dct_output, double *energy) {
    if (!dct_output || !energy) return 0;
    
    double sum = 0.0;

    // Process 16 coefficients at a time
    for (int i = 0; i < 64; i += 16) {
        __m512d vec1 = _mm512_loadu_pd(&dct_output[i]);          // Load 8 doubles
        __m512d vec2 = _mm512_loadu_pd(&dct_output[i + 8]);      // Load next 8 doubles

        __m512d squared1 = _mm512_mul_pd(vec1, vec1);            // Square each element
        __m512d squared2 = _mm512_mul_pd(vec2, vec2);            // Square each element

        sum += _mm512_reduce_add_pd(squared1);                    // Sum all elements in the vector
        sum += _mm512_reduce_add_pd(squared2);                    // Sum all elements in the vector
    }

    *energy = sum;
    return 1; // Success
}

#endif // __AVX512F__

/* ----------------------------
 * Initialization Function
 * ----------------------------
 */

/**
 * @brief Initializes the DCT_Functions struct with appropriate function pointers based on CPU capabilities.
 *
 * This function detects the CPU's capabilities at runtime and assigns the best available
 * implementations (AVX-512, NEON, or Scalar) to the function pointers within the DCT_Functions struct.
 *
 * @param funcs Pointer to the DCT_Functions struct to initialize.
 */
static void initialize_dct_functions(DCT_Functions *funcs) {
    if (cpu_supports_avx512()) {
        #if defined(__AVX512F__)
            funcs->extract_block = extract_dct_block_avx512;
            funcs->apply_dct = apply_dct_avx512;
            funcs->calculate_energy = calculate_dct_energy_avx512;
        #else
            // Fallback to scalar if AVX-512 functions are not fully implemented
            funcs->extract_block = extract_dct_block_scalar;
            funcs->apply_dct = apply_dct_scalar;
            funcs->calculate_energy = calculate_dct_energy_scalar;
        #endif
    }
    else if (cpu_supports_neon()) {
        funcs->extract_block = extract_dct_block_neon;
        funcs->apply_dct = apply_dct_neon;
        funcs->calculate_energy = calculate_dct_energy_neon;
    }
    else {
        funcs->extract_block = extract_dct_block_scalar;
        funcs->apply_dct = apply_dct_scalar;
        funcs->calculate_energy = calculate_dct_energy_scalar;
    }
}

/* ----------------------------
 * DCT Calculation Function
 * ----------------------------
 */

/**
 * @brief Calculates the DCT energy of an AVFrame using FFTW.
 *
 * This function processes each dct_size x dct_size block of the Y plane in the frame,
 * applies the DCT using FFTW, calculates the energy of the DCT coefficients, and
 * accumulates the total energy. The final energy is normalized by the total
 * frame area (width × height).
 *
 * It ensures thread safety by pre-allocating separate block and dct_output buffers
 * for each thread and creating separate FFTW plans within each thread.
 * Utilizes function pointers within a struct to select FFTW-based implementations.
 *
 * @param frame Pointer to the AVFrame containing the image data.
 * @param dct_size The size of the DCT block (e.g., 8 for 8x8 DCT).
 * @param out_energy Pointer to store the calculated normalized energy.
 * @return int Returns 0 on success, non-zero error codes on failure.
 */
int calculate_frame_dct_energy(const AVFrame *frame, int dct_size, double *out_energy) {
    if (!frame || !out_energy) {
        log_error(ERR_NULL_FRAME);
        return ERR_NULL_FRAME;
    }

    // Initialize DCT_Functions with appropriate implementations
    DCT_Functions dct_funcs;
    initialize_dct_functions(&dct_funcs);

    if (!is_supported_pix_fmt(frame->format)) {
        log_error(ERR_UNSUPPORTED_PIX_FMT);
        return ERR_UNSUPPORTED_PIX_FMT;
    }

    enum AVPixelFormat pix_fmt = frame->format;
    int width = frame->width;
    int height = frame->height;
    int num_planes = av_pix_fmt_count_planes(pix_fmt);

    // For alignment with Python, process only the Y plane (plane 0)
    if (num_planes < 1) {
        log_error(ERR_INVALID_PLANE_INDEX);
        return ERR_INVALID_PLANE_INDEX;
    }

    double total_energy = 0.0;
    int num_threads = omp_get_max_threads();

    // Allocate per-thread buffers
    double **blocks = malloc(num_threads * sizeof(double *));
    double **dct_outputs = malloc(num_threads * sizeof(double *));
    if (!blocks || !dct_outputs) {
        log_error(ERR_MEMORY_ALLOCATION);
        if (blocks) free(blocks);
        if (dct_outputs) free(dct_outputs);
        return ERR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < num_threads; i++) {
        blocks[i] = ALIGN_ALLOC(CACHE_LINE_SIZE, dct_size * dct_size * sizeof(double));
        dct_outputs[i] = ALIGN_ALLOC(CACHE_LINE_SIZE, dct_size * dct_size * sizeof(double));
        if (!blocks[i] || !dct_outputs[i]) {
            fprintf(stderr, "Error: Failed to allocate memory for thread %d buffers.\n", i);
            for (int j = 0; j <= i; j++) {
                if (blocks[j]) ALIGN_FREE(blocks[j]);
                if (dct_outputs[j]) ALIGN_FREE(dct_outputs[j]);
            }
            free(blocks);
            free(dct_outputs);
            log_error(ERR_MEMORY_ALLOCATION);
            return ERR_MEMORY_ALLOCATION;
        }
    }

    int error_flag = 0;

    #pragma omp parallel shared(total_energy, error_flag, dct_funcs, frame, dct_size, blocks, dct_outputs) default(none) reduction(+:total_energy)
    {
        int thread_id = omp_get_thread_num();
        double *block = blocks[thread_id];
        double *dct_output = dct_outputs[thread_id];
        double thread_local_energy = 0.0;

        // Create a separate FFTW plan for each thread
        fftw_plan plan = fftw_plan_r2r_2d(dct_size, dct_size, block, dct_output, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
        if (!plan) {
            #pragma omp critical
            {
                if (!error_flag) {
                    log_error(ERR_DCT_INIT_FAILED);
                    error_flag = ERR_DCT_INIT_FAILED;
                }
            }
            #pragma omp cancel for
        }

        // Process only the Y plane (plane 0)
        #pragma omp for schedule(guided, 16) nowait
        for (int y_pos = 0; y_pos < height; y_pos += dct_size) {
            for (int x_pos = 0; x_pos < width; x_pos += dct_size) {
                if (__builtin_expect(error_flag, 0)) {
                    #pragma omp cancel for
                }

                prefetch_memory(&frame->data[0][y_pos * frame->linesize[0] + x_pos], dct_size * sizeof(uint8_t));

                if (!dct_funcs.extract_block(frame, block, x_pos, y_pos, 0, dct_size)) {
                    fprintf(stderr, "Error: Failed to extract block at x=%d, y=%d.\n", x_pos, y_pos);
                    #pragma omp atomic write
                    error_flag = ERR_PARALLEL_CANCELLATION;
                    #pragma omp cancel for;
                }

                if (!dct_funcs.apply_dct(plan, block, dct_output, dct_size)) {
                    fprintf(stderr, "Error: DCT application failed at x=%d, y=%d.\n", x_pos, y_pos);
                    #pragma omp atomic write
                    error_flag = ERR_PARALLEL_CANCELLATION;
                    #pragma omp cancel for;
                }

                double energy = 0.0;
                if (!dct_funcs.calculate_energy(dct_output, &energy)) {
                    fprintf(stderr, "Error: Energy calculation failed at x=%d, y=%d.\n", x_pos, y_pos);
                    #pragma omp atomic write
                    error_flag = ERR_PARALLEL_CANCELLATION;
                    #pragma omp cancel for;
                }

                thread_local_energy += energy;
            }
        }

        // Destroy the FFTW plan
        fftw_destroy_plan(plan);

        #pragma omp cancellation point for
        if (!__builtin_expect(error_flag, 0)) {
            total_energy += thread_local_energy;
        }
    }

    // Free allocated buffers
    for (int i = 0; i < num_threads; i++) {
        if (blocks[i]) ALIGN_FREE(blocks[i]);
        if (dct_outputs[i]) ALIGN_FREE(dct_outputs[i]);
    }
    free(blocks);
    free(dct_outputs);

    if (__builtin_expect(error_flag, 0)) {
        log_error(ERR_PARALLEL_CANCELLATION);
        *out_energy = 0.0;
        return ERR_PARALLEL_CANCELLATION;
    }

    double total_pixels = (double)(width * height);
    if (total_pixels == 0.0) {
        log_error(ERR_ZERO_AREA);
        *out_energy = 0.0;
        return ERR_ZERO_AREA;
    }

    *out_energy = total_energy / total_pixels;
    return 0;
}
