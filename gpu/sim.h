#ifndef SIM_H
#define SIM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include "../transformer/gpu/transformer.h"
#include "../transformer/mlp/gpu/mlp.h"

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

typedef struct {
    // Input projection for scalar pixel -> d_model embedding
    // e_d = x * w_in[d] + b_in[d]
    float* d_in_proj_w;        // [d_model]
    float* d_in_proj_b;        // [d_model]
    float* d_in_proj_w_grad;   // [d_model]
    float* d_in_proj_b_grad;   // [d_model]
    
    // Adam parameters for input projection
    float* d_in_proj_w_m; float* d_in_proj_w_v;
    float* d_in_proj_b_m; float* d_in_proj_b_v;
    float beta1, beta2, epsilon, weight_decay;
    int t; // AdamW time step
    
    // Forward/backward buffers
    float* d_embedded_input;    // [batch_size x seq_len x d_model]
    
    // Diffusion schedule (device arrays, length T)
    int T; // number of diffusion steps
    float* d_betas;                         // [T]
    float* d_alphas;                        // [T]
    float* d_alphas_cumprod;                // [T]
    float* d_sqrt_alphas_cumprod;           // [T]
    float* d_sqrt_one_minus_alphas_cumprod; // [T]
    float* d_sqrt_recip_alphas;             // [T]
    float* d_sqrt_betas;                    // [T]
    
    // Transformer core
    Transformer* transformer;
    
    // Output head: predicts epsilon (noise) per pixel (output_dim = 1)
    MLP* output_mlp;
    
    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
    
    // Dimensions
    int seq_len;     // 784 for 28x28 images
    int d_model;     // e.g. 384
    int batch_size;  // e.g. 4
    int hidden_dim;  // e.g. 1536
    int num_layers;  // e.g. 6
    int image_size;  // 28 for MNIST
    int num_classes; // 10 for MNIST digits
} SIM;

// Function prototypes
SIM* init_sim(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, int T, cublasLtHandle_t cublaslt_handle);
void free_sim(SIM* sim);

// Forward pass: input is noised image x_t (float), class labels, and current timestep index t (1..T)
void forward_pass_sim(SIM* sim, float* d_x_t, unsigned char* d_class_labels, int t_index);

// Loss: MSE between predicted epsilon and true epsilon (noise)
float calculate_loss_sim(SIM* sim, float* d_noise_target);

// Zero gradients for all learnable parameters in SIM (input projection + submodules)
void zero_gradients_sim(SIM* sim);

// Backward pass: uses gradients from output head and transformer to compute d_in_proj gradients
void backward_pass_sim(SIM* sim, float* d_x_t);

// Update weights
void update_weights_sim(SIM* sim, float learning_rate);

// Save/load SIM + submodules
void save_sim(SIM* sim, const char* filename);
SIM* load_sim(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif