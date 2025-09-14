#ifndef SIM_H
#define SIM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../transformer/gpu/transformer.h"
#include "../data.h"

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLASLt Error checking macro
#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// Diffusion model configuration
#define PATCH_SIZE 4                           // 4x4 patches
#define PATCHES_PER_ROW (32 / PATCH_SIZE)      // 8 patches per row/col
#define NUM_PATCHES (PATCHES_PER_ROW * PATCHES_PER_ROW)  // 64 total patches
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * 3) // 48 (4x4x3 RGB values)
#define MAX_TIMESTEPS 1000
#define MIN_BETA 0.0001f
#define MAX_BETA 0.02f

typedef struct {
    // Patch embedding layer (linear projection)
    float* d_patch_projection;      // [PATCH_DIM x d_model]
    float* d_patch_projection_grad; // [PATCH_DIM x d_model]
    
    // Timestep embedding layer
    float* d_time_embedding;        // [MAX_TIMESTEPS x d_model]
    float* d_time_embedding_grad;   // [MAX_TIMESTEPS x d_model]
    
    // Adam parameters for embeddings
    float* d_patch_projection_m, *d_patch_projection_v;
    float* d_time_embedding_m, *d_time_embedding_v;
    float beta1, beta2, epsilon, weight_decay;
    int t;
    
    // Forward pass buffers
    float* d_patches;               // [batch_size x NUM_PATCHES x PATCH_DIM]
    float* d_patch_embeds;          // [batch_size x NUM_PATCHES x d_model]
    float* d_time_embeds;           // [batch_size x d_model]
    float* d_input_embeds;          // [batch_size x (NUM_PATCHES+1) x d_model]
    float* d_noise_pred;            // [batch_size x NUM_PATCHES x PATCH_DIM]
    
    // Loss computation buffer
    float* d_loss_result;           // [1]
    
    // Noise schedule
    float* h_betas;                 // [MAX_TIMESTEPS] - noise schedule on host
    float* h_alphas;                // [MAX_TIMESTEPS] - 1 - betas
    float* h_alpha_bars;            // [MAX_TIMESTEPS] - cumulative product of alphas
    
    // Random number generation
    curandState* d_curand_states;   // [batch_size * NUM_PATCHES]
    
    // Transformer core
    Transformer* transformer;
    
    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
    
    // Dimensions
    int d_model;
    int hidden_dim;
    int num_layers;
    int batch_size;
    int seq_len;  // NUM_PATCHES + 1 (for time embedding)
} SIM;

// Function prototypes
SIM* init_diffusion_model(int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_diffusion_model(SIM* model);

// Core diffusion operations
void image_to_patches(float* d_patches, float* d_images, int batch_size);
void patches_to_image(float* d_images, float* d_patches, int batch_size);
void forward_diffusion(SIM* model, float* d_noisy_patches, float* d_clean_patches, int* timesteps, int batch_size);
void reverse_diffusion_step(SIM* model, float* d_patches, int timestep, int batch_size);

// Training operations
void forward_pass_diffusion(SIM* model, float* d_patches, int* timesteps);
float calculate_loss_diffusion(SIM* model, float* d_target_noise);
void zero_gradients_diffusion(SIM* model);
void backward_pass_diffusion(SIM* model, float* d_patches, int* timesteps);
void update_weights_diffusion(SIM* model, float learning_rate);

// Sampling
void sample_images(SIM* model, float* d_output_images, int batch_size);

// Utility functions
void save_diffusion_model(SIM* model, const char* filename);
SIM* load_diffusion_model(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);
void cifar10_to_float(float* output, const CIFAR10_Image* images, int num_images);

#endif