#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <jpeglib.h>
#include "ssm/gpu/ssm.h"

// We'll dynamically determine this now
#define NUM_NOISE_STEPS 1024

// Image structure
typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// Structure for Hilbert mapping
typedef struct {
    int *to_1d;
    int *to_2d;
    int width;
    int height;
} HilbertMap;

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
}

// Find the next power of 2
int next_power_of_2(int n) {
    int power = 1;
    while (power < n) power *= 2;
    return power;
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
HilbertMap create_hilbert_mapping(int width, int height) {
    HilbertMap map;
    map.width = width;
    map.height = height;
    
    // Determine n (power of 2 >= max(width, height))
    int n = next_power_of_2((width > height) ? width : height);
    
    // Allocate memory for map arrays
    map.to_1d = (int*)malloc(width * height * sizeof(int));
    map.to_2d = (int*)malloc(width * height * sizeof(int));
    
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
        map.to_1d[orig_idx] = i;            // Maps from 2D to 1D
        map.to_2d[i] = orig_idx;            // Maps from 1D to 2D
    }
    
    free(points);
    return map;
}

// Generate pure noise image with values between 0-1
void generate_pure_noise(float *data, int length) {
    for (int i = 0; i < length; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Convert flat array back to image using Hilbert mapping
unsigned char* unflatten_with_hilbert(float *flattened, HilbertMap map) {
    unsigned char *unflattened = (unsigned char*)malloc(map.width * map.height);
    
    for (int i = 0; i < map.width * map.height; i++) {
        // Convert from normalized float to byte
        unflattened[map.to_2d[i]] = (unsigned char)(flattened[i] * 255.0f);
    }
    
    return unflattened;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Fixed parameters (except for image_size which will be determined from model)
    int steps = NUM_NOISE_STEPS;
    char output_prefix[256] = "denoised";
    
    // Files for stacked models
    char *layer1_filename = NULL;
    char *layer2_filename = NULL;
    char *layer3_filename = NULL;
    char *layer4_filename = NULL;
    
    // Check if correct number of arguments
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <layer1_model> <layer2_model> <layer3_model> <layer4_model>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    // Get model filenames
    layer1_filename = argv[1];
    layer2_filename = argv[2];
    layer3_filename = argv[3];
    layer4_filename = argv[4];
    
    printf("=== SIGM Image Generation ===\n");
    printf("Layer 1 model: %s\n", layer1_filename);
    printf("Layer 2 model: %s\n", layer2_filename);
    printf("Layer 3 model: %s\n", layer3_filename);
    printf("Layer 4 model: %s\n", layer4_filename);
    
    // Load models
    SSM *layer1_ssm = load_ssm(layer1_filename, 1);
    SSM *layer2_ssm = load_ssm(layer2_filename, 1);
    SSM *layer3_ssm = load_ssm(layer3_filename, 1);
    SSM *layer4_ssm = load_ssm(layer4_filename, 1);
    
    // Determine image dimensions from model input size
    int flattened_length = layer1_ssm->input_dim;
    // Calculate image size (square root of flattened length)
    int image_size = (int)sqrtf((float)flattened_length);
    
    printf("Model dimensions:\n");
    printf("Layer 1: input_dim=%d, output_dim=%d, state_dim=%d\n", 
           layer1_ssm->input_dim, layer1_ssm->output_dim, layer1_ssm->state_dim);
    printf("Layer 2: input_dim=%d, output_dim=%d, state_dim=%d\n", 
           layer2_ssm->input_dim, layer2_ssm->output_dim, layer2_ssm->state_dim);
    printf("Layer 3: input_dim=%d, output_dim=%d, state_dim=%d\n", 
           layer3_ssm->input_dim, layer3_ssm->output_dim, layer3_ssm->state_dim);
    printf("Layer 4: input_dim=%d, output_dim=%d, state_dim=%d\n", 
           layer4_ssm->input_dim, layer4_ssm->output_dim, layer4_ssm->state_dim);
    printf("Detected image size: %dx%d (%d pixels)\n", 
           image_size, image_size, flattened_length);
    printf("Denoising steps: %d\n", steps);
    printf("Output prefix: %s\n\n", output_prefix);
    
    // Verify that the model dimensions make sense for a square image
    if (image_size * image_size != flattened_length) {
        fprintf(stderr, "Error: Model input dimension (%d) is not a perfect square\n", 
                flattened_length);
        return EXIT_FAILURE;
    }
    
    // Verify that layer4 output matches layer1 input
    if (layer4_ssm->output_dim != layer1_ssm->input_dim) {
        fprintf(stderr, "Error: Layer 4 output dimension (%d) doesn't match Layer 1 input dimension (%d)\n",
                layer4_ssm->output_dim, layer1_ssm->input_dim);
        return EXIT_FAILURE;
    }
    
    // Create the Hilbert mapping with the correct image size
    HilbertMap hilbert_map = create_hilbert_mapping(image_size, image_size);
    
    // Reset SSM states
    CHECK_CUDA(cudaMemset(layer1_ssm->d_state, 0, layer1_ssm->batch_size * layer1_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer2_ssm->d_state, 0, layer2_ssm->batch_size * layer2_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer3_ssm->d_state, 0, layer3_ssm->batch_size * layer3_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer4_ssm->d_state, 0, layer4_ssm->batch_size * layer4_ssm->state_dim * sizeof(float)));
    
    // Allocate memory
    float *h_current_image = (float*)malloc(flattened_length * sizeof(float));
    float *d_input;
    float *d_layer1_output;
    float *d_layer2_output;
    float *d_layer3_output;
    
    printf("Allocating GPU memory...\n");
    CHECK_CUDA(cudaMalloc(&d_input, flattened_length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer1_output, layer1_ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_output, layer2_ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_output, layer3_ssm->output_dim * sizeof(float)));
    
    // Generate pure noise
    printf("Generating initial noise...\n");
    generate_pure_noise(h_current_image, flattened_length);
    
    // Save the initial noise image
    unsigned char *noise_image = unflatten_with_hilbert(h_current_image, hilbert_map);
    char initial_filename[512];
    sprintf(initial_filename, "%s_initial_noise.jpg", output_prefix);
    save_grayscale_image(initial_filename, noise_image, image_size, image_size);
    free(noise_image);
    printf("Saved initial noise as %s\n", initial_filename);
    
    // Copy the noise to GPU
    printf("Copying initial noise to GPU (size: %d floats)...\n", flattened_length);
    CHECK_CUDA(cudaMemcpy(d_input, h_current_image, flattened_length * sizeof(float), cudaMemcpyHostToDevice));
    
    // Denoise the image step by step
    printf("Starting denoising process with %d steps...\n", steps);
    
    // We'll save a few intermediate steps
    int save_steps[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, steps-1};
    int num_save_steps = sizeof(save_steps) / sizeof(save_steps[0]);
    
    for (int step = 0; step < steps; step++) {
        // Forward pass through layer 1 model
        forward_pass(layer1_ssm, d_input);
        
        // Copy layer1 output for layer2 input
        CHECK_CUDA(cudaMemcpy(d_layer1_output, layer1_ssm->d_predictions, 
                           layer1_ssm->output_dim * sizeof(float), 
                           cudaMemcpyDeviceToDevice));
        
        // Forward pass through layer 2 model
        forward_pass(layer2_ssm, d_layer1_output);
        
        // Copy layer2 output for layer3 input
        CHECK_CUDA(cudaMemcpy(d_layer2_output, layer2_ssm->d_predictions, 
                           layer2_ssm->output_dim * sizeof(float), 
                           cudaMemcpyDeviceToDevice));
        
        // Forward pass through layer 3 model
        forward_pass(layer3_ssm, d_layer2_output);
        
        // Copy layer3 output for layer4 input
        CHECK_CUDA(cudaMemcpy(d_layer3_output, layer3_ssm->d_predictions, 
                           layer3_ssm->output_dim * sizeof(float), 
                           cudaMemcpyDeviceToDevice));
        
        // Forward pass through layer 4 model
        forward_pass(layer4_ssm, d_layer3_output);
        
        // Copy output back to input for next iteration
        CHECK_CUDA(cudaMemcpy(d_input, layer4_ssm->d_predictions, 
                           flattened_length * sizeof(float), 
                           cudaMemcpyDeviceToDevice));
        
        // Print progress every few steps
        if (step == 0 || step == steps-1 || step % (steps/10) == 0) {
            printf("Denoising step %d/%d completed\n", step+1, steps);
        }
        
        // Save intermediate results at specified steps
        for (int i = 0; i < num_save_steps; i++) {
            if (step == save_steps[i]) {
                // Copy current denoised image to host
                CHECK_CUDA(cudaMemcpy(h_current_image, layer4_ssm->d_predictions, 
                                   flattened_length * sizeof(float), 
                                   cudaMemcpyDeviceToHost));
                
                // Convert to image and save
                unsigned char *denoised_image = unflatten_with_hilbert(h_current_image, hilbert_map);
                char step_filename[512];
                sprintf(step_filename, "%s_step_%04d.jpg", output_prefix, step);
                save_grayscale_image(step_filename, denoised_image, image_size, image_size);
                free(denoised_image);
                printf("Saved intermediate result at step %d as %s\n", step, step_filename);
            }
        }
    }
    
    // Copy final denoised image to host
    CHECK_CUDA(cudaMemcpy(h_current_image, layer4_ssm->d_predictions, 
                       flattened_length * sizeof(float), 
                       cudaMemcpyDeviceToHost));
    
    // Convert to image and save final result
    unsigned char *final_image = unflatten_with_hilbert(h_current_image, hilbert_map);
    char final_filename[512];
    sprintf(final_filename, "%s_final.jpg", output_prefix);
    save_grayscale_image(final_filename, final_image, image_size, image_size);
    free(final_image);
    
    printf("\nDenoising process complete!\n");
    printf("Final image saved as: %s\n", final_filename);
    
    // Clean up
    free(h_current_image);
    free(hilbert_map.to_1d);
    free(hilbert_map.to_2d);
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
    cudaFree(d_input);
    cudaFree(d_layer1_output);
    cudaFree(d_layer2_output);
    cudaFree(d_layer3_output);
    
    return 0;
}