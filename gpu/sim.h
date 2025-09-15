#ifndef SIM_H
#define SIM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../transformer/gpu/transformer.h"
#include "../transformer/mlp/gpu/mlp.h"
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

// SIM configuration
#define PATCH_SIZE 4                           // 4x4 patches
#define PATCHES_PER_ROW (32 / PATCH_SIZE)      // 8 patches per row/col
#define NUM_PATCHES (PATCHES_PER_ROW * PATCHES_PER_ROW)  // 64 total patches
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * 3) // 48 (4x4x3 RGB values)
#define MAX_TIMESTEPS 1000
#define MIN_BETA 0.0001f
#define MAX_BETA 0.02f

typedef struct {
    // Time step embedding layer
    float* d_time_embedding;      // [MAX_TIMESTEPS x d_model]
    float* d_time_embedding_grad; // [MAX_TIMESTEPS x d_model]
    
    // Adam parameters for time embeddings
    float* d_time_embedding_m, *d_time_embedding_v;
    float beta1, beta2, epsilon, weight_decay;
    int t;
    
    // Forward pass buffers
    float* d_patches;          // [batch_size x NUM_PATCHES x PATCH_DIM]
    float* d_noisy_patches;    // [batch_size x NUM_PATCHES x PATCH_DIM]
    float* d_noise;            // [batch_size x NUM_PATCHES x PATCH_DIM]
    float* d_patch_embeds;     // [batch_size x NUM_PATCHES x d_model]
    
    // Loss computation buffer
    float* d_loss_result;      // [1]
    
    // Noise schedule
    float* h_betas;            // [MAX_TIMESTEPS] - noise schedule on host
    float* h_alphas;           // [MAX_TIMESTEPS] - 1 - betas
    float* h_alpha_bars;       // [MAX_TIMESTEPS] - cumulative product of alphas
    
    // Random number generation
    curandState* d_curand_states;
    
    // Transformer core
    Transformer* transformer;
    
    // Input projection MLP (PATCH_DIM -> d_model)
    MLP* input_mlp;
    
    // Output projection MLP (d_model -> PATCH_DIM) 
    MLP* output_mlp;
    
    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
    
    // Dimensions
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
} SIM;

// Function prototypes
SIM* init_sim(int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_sim(SIM* sim);
void forward_pass_sim(SIM* sim, float* d_clean_images, int timestep);
float calculate_loss_sim(SIM* sim);
void zero_gradients_sim(SIM* sim);
void backward_pass_sim(SIM* sim, int timestep);
void update_weights_sim(SIM* sim, float learning_rate);
void save_sim(SIM* sim, const char* filename);
SIM* load_sim(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif