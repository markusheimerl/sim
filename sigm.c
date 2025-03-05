#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jpeglib.h>
#include <time.h>

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// Function to load a JPEG image
Image load_jpeg(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;
    Image img = {0};
    
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return img;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    
    img.width = cinfo.output_width;
    img.height = cinfo.output_height;
    img.channels = cinfo.output_components;
    img.data = (unsigned char *)malloc(img.width * img.height * img.channels);
    
    row_stride = img.width * img.channels;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(img.data + (cinfo.output_scanline - 1) * row_stride, buffer[0], row_stride);
    }
    
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return img;
}

// Function to save a grayscale image
void save_grayscale_image(const char *filename, unsigned char *data, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPARRAY buffer;
    
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;
    
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width, 1);
    
    while (cinfo.next_scanline < cinfo.image_height) {
        memcpy(buffer[0], data + cinfo.next_scanline * width, width);
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    
    printf("Saved %s\n", filename);
}

// Function to save a 1D array as a multi-row JPEG to handle dimension limits
void save_1d_as_jpeg(const char *filename, unsigned char *data, int length) {
    // JPEG has limitations on max dimension, so arrange in multiple rows if needed
    // Maximum width for JPEG is usually around 65,000 pixels
    const int MAX_WIDTH = 4096;  // Use a more reasonable limit
    
    int width, height;
    if (length <= MAX_WIDTH) {
        width = length;
        height = 1;
    } else {
        width = MAX_WIDTH;
        height = (length + MAX_WIDTH - 1) / MAX_WIDTH;  // Ceiling division
    }
    
    // Create a 2D array from our 1D data
    unsigned char *img_data = (unsigned char *)malloc(width * height);
    memset(img_data, 0, width * height);  // Initialize to 0
    
    // Copy the data
    for (int i = 0; i < length; i++) {
        int row = i / width;
        int col = i % width;
        img_data[row * width + col] = data[i];
    }
    
    // Save as JPEG
    save_grayscale_image(filename, img_data, width, height);
    free(img_data);
    
    printf("Saved 1D data as JPEG: %s (%d×%d)\n", filename, width, height);
}

// Scale an image to a new size using nearest neighbor interpolation
Image scale_image(Image src, int new_width, int new_height) {
    Image dst = {0};
    dst.width = new_width;
    dst.height = new_height;
    dst.channels = src.channels;
    dst.data = (unsigned char *)malloc(new_width * new_height * src.channels);
    
    double x_ratio = (double)src.width / new_width;
    double y_ratio = (double)src.height / new_height;
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);
            
            // Clamp to image bounds
            if (src_x >= src.width) src_x = src.width - 1;
            if (src_y >= src.height) src_y = src.height - 1;
            
            for (int c = 0; c < src.channels; c++) {
                int src_idx = (src_y * src.width + src_x) * src.channels + c;
                int dst_idx = (y * new_width + x) * src.channels + c;
                dst.data[dst_idx] = src.data[src_idx];
            }
        }
    }
    
    printf("Scaled image from %d×%d to %d×%d\n", src.width, src.height, new_width, new_height);
    return dst;
}

// Calculate scaling factors to resize an image to a maximum dimension
void calculate_scaling_factors(int width, int height, int max_dim, int *new_width, int *new_height) {
    if (width <= max_dim && height <= max_dim) {
        // Image is already small enough
        *new_width = width;
        *new_height = height;
    } else if (width > height) {
        // Width is the larger dimension
        *new_width = max_dim;
        *new_height = (int)(height * ((double)max_dim / width));
    } else {
        // Height is the larger dimension
        *new_height = max_dim;
        *new_width = (int)(width * ((double)max_dim / height));
    }
}

// Add Gaussian noise with specified standard deviation to a normalized float array
void add_gaussian_noise_float(float *data, int length, float noise_level) {
    for (int i = 0; i < length; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        
        // Avoid log(0)
        if (u1 < 1e-8) u1 = 1e-8;
        
        // Box-Muller transform
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        
        // Add noise to normalized data
        data[i] += (float)(z0 * noise_level);
        
        // Clamp values between 0 and 1
        if (data[i] < 0.0f) data[i] = 0.0f;
        if (data[i] > 1.0f) data[i] = 1.0f;
    }
}

// Generate pure noise image with values between 0-1
void generate_pure_noise(float *data, int length) {
    for (int i = 0; i < length; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// Blend between source image and target image with given factor (0.0 = source, 1.0 = target)
void blend_images(float *result, float *source, float *target, int length, float blend_factor) {
    for (int i = 0; i < length; i++) {
        result[i] = source[i] * (1.0f - blend_factor) + target[i] * blend_factor;
    }
}

// Extract center square from an image
Image extract_center_square(Image img) {
    Image square = {0};
    
    // Determine the size of the square (minimum of width and height)
    int size = (img.width < img.height) ? img.width : img.height;
    
    // Calculate top-left corner of the square
    int start_x = (img.width - size) / 2;
    int start_y = (img.height - size) / 2;
    
    // Create the square image
    square.width = size;
    square.height = size;
    square.channels = img.channels;
    square.data = (unsigned char *)malloc(size * size * img.channels);
    
    // Copy the center square from the original image
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            for (int c = 0; c < img.channels; c++) {
                int src_idx = ((start_y + y) * img.width + (start_x + x)) * img.channels + c;
                int dst_idx = (y * size + x) * img.channels + c;
                square.data[dst_idx] = img.data[src_idx];
            }
        }
    }
    
    printf("Extracted center square: %d x %d\n", size, size);
    return square;
}

// Convert color image to grayscale
unsigned char* convert_to_grayscale(Image img) {
    unsigned char *grayscale = (unsigned char *)malloc(img.width * img.height);
    
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = (y * img.width + x) * img.channels;
            
            if (img.channels >= 3) {
                // Standard luminance formula
                grayscale[y * img.width + x] = (unsigned char)(
                    0.299 * img.data[idx] +      // Red
                    0.587 * img.data[idx + 1] +  // Green
                    0.114 * img.data[idx + 2]);  // Blue
            } else {
                // Already grayscale
                grayscale[y * img.width + x] = img.data[idx];
            }
        }
    }
    
    return grayscale;
}

// Rotate/flip a quadrant appropriately for Hilbert curve
void rotate(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }
        // Swap x and y
        int t = *x;
        *x = *y;
        *y = t;
    }
}

// Convert (x,y) to d (distance along Hilbert curve)
int xy2d(int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rotate(s, &x, &y, rx, ry);
    }
    return d;
}

// Convert d (distance along Hilbert curve) to (x,y)
void d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t=d;
    *x = *y = 0;
    for (s=1; s<n; s*=2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rotate(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

// Find the next power of 2
int next_power_of_2(int n) {
    int power = 1;
    while (power < n) power *= 2;
    return power;
}

// Structure to hold 2D coordinates and Hilbert indices
typedef struct {
    int x, y;      // 2D coordinates
    int hilbert_d; // Hilbert distance
} HilbertPoint;

// Comparison function for qsort
int compare_hilbert(const void *a, const void *b) {
    return ((HilbertPoint*)a)->hilbert_d - ((HilbertPoint*)b)->hilbert_d;
}

// Create a mapping from 2D to 1D using Hilbert curve ordering
void create_hilbert_mapping(int width, int height, int **to_1d, int **to_2d) {
    // Determine n (power of 2 >= max(width, height))
    int n = next_power_of_2((width > height) ? width : height);
    
    // Allocate memory for map arrays
    *to_1d = (int*)malloc(width * height * sizeof(int));
    *to_2d = (int*)malloc(width * height * sizeof(int));
    
    // Create array of points with their Hilbert distances
    HilbertPoint *points = (HilbertPoint*)malloc(width * height * sizeof(HilbertPoint));
    int idx = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            points[idx].x = x;
            points[idx].y = y;
            points[idx].hilbert_d = xy2d(n, x, y);
            idx++;
        }
    }
    
    // Sort points by Hilbert distance
    qsort(points, width * height, sizeof(HilbertPoint), compare_hilbert);
    
    // Fill in the mapping arrays
    for (int i = 0; i < width * height; i++) {
        int orig_idx = points[i].y * width + points[i].x;
        (*to_1d)[orig_idx] = i;  // Maps from 2D to 1D
        (*to_2d)[i] = orig_idx;  // Maps from 1D to 2D
    }
    
    free(points);
}

// Flatten a grayscale image using Hilbert curve ordering
float* flatten_with_hilbert_float(unsigned char *grayscale, int width, int height, int **to_2d) {
    // Create forward and backward mappings
    int *to_1d;
    create_hilbert_mapping(width, height, &to_1d, to_2d);
    
    // Create the flattened array as float (normalized 0-1)
    float *flattened = (float*)malloc(width * height * sizeof(float));
    
    // Flatten using the mapping and normalize to 0-1
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int orig_idx = y * width + x;
            int flat_idx = to_1d[orig_idx];
            flattened[flat_idx] = grayscale[orig_idx] / 255.0f;
        }
    }
    
    free(to_1d);
    return flattened;
}

// Unflatten a 1D array back to a 2D image
unsigned char* unflatten_with_hilbert(float *flattened, int width, int height, int *to_2d) {
    unsigned char *unflattened = (unsigned char*)malloc(width * height);
    
    for (int i = 0; i < width * height; i++) {
        // Convert back from normalized float to byte
        unflattened[to_2d[i]] = (unsigned char)(flattened[i] * 255.0f);
    }
    
    return unflattened;
}

// Write a float array to a CSV file as a row
void write_float_array_to_csv(FILE *fp, float *array, int length) {
    for (int i = 0; i < length; i++) {
        fprintf(fp, "%.6f", array[i]);
        if (i < length - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");
}

int main() {
    // Seed the random number generator
    srand(time(NULL));
    
    const char *input_file = "img.jpg";
    const int NUM_NOISE_STEPS = 1024;
    
    // Load the image
    Image img = load_jpeg(input_file);
    if (img.data == NULL) {
        fprintf(stderr, "Failed to load image: %s\n", input_file);
        return 1;
    }
    
    printf("Original image loaded: %d x %d with %d channels\n", img.width, img.height, img.channels);
    
    // Step 1: Scale down the image to a maximum dimension
    int max_dimension = 128; // Use 128 as requested
    int new_width, new_height;
    calculate_scaling_factors(img.width, img.height, max_dimension, &new_width, &new_height);
    
    Image scaled_img;
    if (new_width != img.width || new_height != img.height) {
        scaled_img = scale_image(img, new_width, new_height);
        free(img.data); // Free the original image data
    } else {
        printf("No scaling needed, dimensions already appropriate\n");
        scaled_img = img; // Just use the original
    }
    
    // Step 2: Extract center square from the scaled image
    Image square_img = extract_center_square(scaled_img);
    free(scaled_img.data); // Free the scaled image data
    
    // Step 3: Convert to grayscale
    unsigned char *grayscale = convert_to_grayscale(square_img);
    
    // Save the grayscale image
    save_grayscale_image("grayscale.jpg", grayscale, square_img.width, square_img.height);
    
    // Step 4: Create Hilbert mapping and flatten the image
    int *to_2d;
    float *clean_flattened = flatten_with_hilbert_float(grayscale, square_img.width, square_img.height, &to_2d);
    
    // Get the length of the flattened array
    int flattened_length = square_img.width * square_img.height;
    
    // Create a CSV file for the dataset
    char csv_filename[256];
    sprintf(csv_filename, "denoising_dataset_%dx%d.csv", square_img.width, square_img.height);
    FILE *csv_file = fopen(csv_filename, "w");
    if (!csv_file) {
        fprintf(stderr, "Failed to create CSV file %s\n", csv_filename);
        return 1;
    }
    
    // Allocate an array to hold all the noisy versions
    float **noisy_arrays = (float**)malloc(NUM_NOISE_STEPS * sizeof(float*));
    
    // Create a pure noise image for the most noisy step
    float *pure_noise = (float*)malloc(flattened_length * sizeof(float));
    generate_pure_noise(pure_noise, flattened_length);
    
    // Create increasingly noisy versions of the image
    printf("Generating %d noise steps...\n", NUM_NOISE_STEPS);
    
    // For visualization, save a few stages
    int vis_steps[] = {0, 1, 4, 16, 64, 256, 512, 768, 1022, 1023};
    int num_vis_steps = sizeof(vis_steps) / sizeof(vis_steps[0]);
    
    for (int i = 0; i < NUM_NOISE_STEPS; i++) {
        // Allocate memory for this noise step
        noisy_arrays[i] = (float*)malloc(flattened_length * sizeof(float));
        
        // Calculate the blend factor (0 = clean image, 1 = pure noise)
        // Use a more aggressive curve to ensure the highest noise step is almost pure noise
        float blend_factor = (float)i / (NUM_NOISE_STEPS - 1);
        
        // Apply a power curve to make higher noise levels even more noisy
        // The exponent 0.5 creates a curve that gets to high noise levels faster
        blend_factor = powf(blend_factor, 0.5f);
        
        // For the last few steps, ensure we reach pure noise
        if (i >= NUM_NOISE_STEPS - 5) {
            // Blend quickly to pure noise in the last 5 steps
            float last_steps_factor = (float)(i - (NUM_NOISE_STEPS - 5)) / 4.0f;
            blend_factor = blend_factor * (1.0f - last_steps_factor) + last_steps_factor;
        }
        
        // Use the blend factor to interpolate between clean image and pure noise
        blend_images(noisy_arrays[i], clean_flattened, pure_noise, flattened_length, blend_factor);
        
        // Add additional random noise to avoid too smooth transitions
        float noise_level = 0.05f * blend_factor; // Scale noise with blend factor
        add_gaussian_noise_float(noisy_arrays[i], flattened_length, noise_level);
        
        // Make the final step pure noise
        if (i == NUM_NOISE_STEPS - 1) {
            memcpy(noisy_arrays[i], pure_noise, flattened_length * sizeof(float));
        }
        
        // Visualize select noise levels
        for (int v = 0; v < num_vis_steps; v++) {
            if (i == vis_steps[v]) {
                // Convert back to image and save for visualization
                unsigned char *noisy_image = unflatten_with_hilbert(noisy_arrays[i], 
                                                                   square_img.width, square_img.height, to_2d);
                char vis_filename[256];
                sprintf(vis_filename, "noisy_%d_of_%d.jpg", i, NUM_NOISE_STEPS-1);
                save_grayscale_image(vis_filename, noisy_image, square_img.width, square_img.height);
                free(noisy_image);
                
                printf("Saved visualization for noise step %d (blend factor: %.4f)\n", 
                       i, blend_factor);
            }
        }
        
        // Show progress
        if (i % 100 == 0 || i == NUM_NOISE_STEPS - 1) {
            printf("Generated noise step %d/%d (blend factor: %.4f)\n", 
                   i+1, NUM_NOISE_STEPS, blend_factor);
        }
    }
    
    // Write the CSV file in reverse order (from most noisy to clean)
    printf("Writing CSV file with %d rows...\n", NUM_NOISE_STEPS);
    for (int i = NUM_NOISE_STEPS - 1; i >= 0; i--) {
        write_float_array_to_csv(csv_file, noisy_arrays[i], flattened_length);
        
        // Show progress
        if (i % 100 == 0 || i == 0) {
            printf("Wrote row %d/%d to CSV\n", NUM_NOISE_STEPS - i, NUM_NOISE_STEPS);
        }
    }
    
    // Close the CSV file
    fclose(csv_file);
    printf("CSV dataset created: %s\n", csv_filename);
    
    // Clean up all noisy arrays
    for (int i = 0; i < NUM_NOISE_STEPS; i++) {
        free(noisy_arrays[i]);
    }
    free(noisy_arrays);
    free(pure_noise);
    
    // Clean up other allocations
    free(square_img.data);
    free(grayscale);
    free(clean_flattened);
    free(to_2d);
    
    return 0;
}