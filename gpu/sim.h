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

#define PATCH_SIZE 8
#define PATCHES_PER_ROW (32 / PATCH_SIZE)
#define NUM_PATCHES (PATCHES_PER_ROW * PATCHES_PER_ROW)
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * 3)
#define MAX_TIMESTEPS 1000
#define MIN_BETA 0.0001f
#define MAX_BETA 0.02f

typedef struct {
    // Time embeddings (learnable)
    float* d_time_embedding;
    float* d_time_embedding_grad;
    float* d_time_embedding_m;
    float* d_time_embedding_v;
    
    // Forward pass buffers
    float* d_clean_patches;
    float* d_noisy_patches;
    float* d_noise;
    float* d_patch_embeds;
    float* d_loss_result;
    
    // Noise schedule (host)
    float* h_alpha_bars;
    
    // Random states
    curandState* d_curand_states;
    
    // Core components
    Transformer* transformer;
    MLP* input_mlp;
    MLP* output_mlp;
    
    // Parameters
    float beta1, beta2, epsilon, weight_decay;
    int t;
    int d_model, batch_size, hidden_dim, num_layers;
    cublasLtHandle_t cublaslt_handle;
} SIM;

// Function prototypes
SIM* init_sim(int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_sim(SIM* sim);
void forward_pass_sim(SIM* sim, float* d_images, int timestep);
float calculate_loss_sim(SIM* sim);
void zero_gradients_sim(SIM* sim);
void backward_pass_sim(SIM* sim, int timestep);
void update_weights_sim(SIM* sim, float learning_rate);
void save_sim(SIM* sim, const char* filename);
SIM* load_sim(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif