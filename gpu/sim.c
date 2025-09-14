#include "sim.h"

// Initialize curand states
__global__ static void init_curand_states(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Convert image to patches
__global__ static void image_to_patches_kernel(float* patches, float* images, int batch_size) {
    int b = blockIdx.x;
    int patch_idx = blockIdx.y;
    int pixel_in_patch = threadIdx.x;
    
    if (b >= batch_size || patch_idx >= NUM_PATCHES || pixel_in_patch >= PATCH_DIM) return;
    
    // Calculate patch position
    int patch_row = patch_idx / PATCHES_PER_ROW;
    int patch_col = patch_idx % PATCHES_PER_ROW;
    
    // Calculate pixel position within patch
    int pixel_idx = pixel_in_patch / 3;  // Which pixel in the 4x4 patch
    int channel = pixel_in_patch % 3;    // Which color channel
    
    int pixel_row = pixel_idx / PATCH_SIZE;
    int pixel_col = pixel_idx % PATCH_SIZE;
    
    // Global pixel coordinates
    int global_row = patch_row * PATCH_SIZE + pixel_row;
    int global_col = patch_col * PATCH_SIZE + pixel_col;
    
    // Source index in image (RGB interleaved format)
    int img_idx = b * (32 * 32 * 3) + (global_row * 32 + global_col) * 3 + channel;
    
    // Destination index in patches
    int patch_output_idx = b * NUM_PATCHES * PATCH_DIM + patch_idx * PATCH_DIM + pixel_in_patch;
    
    patches[patch_output_idx] = images[img_idx];
}

// Convert patches back to image
__global__ static void patches_to_image_kernel(float* images, float* patches, int batch_size) {
    int b = blockIdx.x;
    int patch_idx = blockIdx.y;
    int pixel_in_patch = threadIdx.x;
    
    if (b >= batch_size || patch_idx >= NUM_PATCHES || pixel_in_patch >= PATCH_DIM) return;
    
    // Calculate patch position
    int patch_row = patch_idx / PATCHES_PER_ROW;
    int patch_col = patch_idx % PATCHES_PER_ROW;
    
    // Calculate pixel position within patch
    int pixel_idx = pixel_in_patch / 3;  // Which pixel in the 4x4 patch
    int channel = pixel_in_patch % 3;    // Which color channel
    
    int pixel_row = pixel_idx / PATCH_SIZE;
    int pixel_col = pixel_idx % PATCH_SIZE;
    
    // Global pixel coordinates
    int global_row = patch_row * PATCH_SIZE + pixel_row;
    int global_col = patch_col * PATCH_SIZE + pixel_col;
    
    // Destination index in image (RGB interleaved format)
    int img_idx = b * (32 * 32 * 3) + (global_row * 32 + global_col) * 3 + channel;
    
    // Source index in patches
    int patch_input_idx = b * NUM_PATCHES * PATCH_DIM + patch_idx * PATCH_DIM + pixel_in_patch;
    
    images[img_idx] = patches[patch_input_idx];
}

// Add noise to clean patches
__global__ static void add_noise_kernel(float* noisy_patches, float* clean_patches, float* noise,
                                       float sqrt_alpha_bar, float sqrt_one_minus_alpha_bar,
                                       int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        noisy_patches[idx] = sqrt_alpha_bar * clean_patches[idx] + sqrt_one_minus_alpha_bar * noise[idx];
    }
}

// Generate random noise
__global__ static void generate_noise_kernel(float* noise, curandState* states, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        noise[idx] = curand_normal(&states[idx]);
    }
}

// Initialize the diffusion model
DiffusionModel* init_diffusion_model(int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    DiffusionModel* model = (DiffusionModel*)malloc(sizeof(DiffusionModel));
    
    // Store dimensions
    model->d_model = d_model;
    model->hidden_dim = hidden_dim;
    model->num_layers = num_layers;
    model->batch_size = batch_size;
    model->seq_len = NUM_PATCHES + 1;  // patches + time embedding
    model->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    model->beta1 = 0.9f;
    model->beta2 = 0.999f;
    model->epsilon = 1e-8f;
    model->t = 0;
    model->weight_decay = 0.01f;
    
    int patch_proj_size = PATCH_DIM * d_model;
    int time_emb_size = MAX_TIMESTEPS * d_model;
    int patches_size = batch_size * NUM_PATCHES * PATCH_DIM;
    int patch_embeds_size = batch_size * NUM_PATCHES * d_model;
    int time_embeds_size = batch_size * d_model;
    int input_embeds_size = batch_size * model->seq_len * d_model;
    
    // Allocate device memory for embeddings
    CHECK_CUDA(cudaMalloc(&model->d_patch_projection, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_patch_projection_grad, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_time_embedding, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_time_embedding_grad, time_emb_size * sizeof(float)));
    
    // Allocate Adam parameters
    CHECK_CUDA(cudaMalloc(&model->d_patch_projection_m, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_patch_projection_v, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_time_embedding_m, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_time_embedding_v, time_emb_size * sizeof(float)));
    
    // Allocate forward pass buffers
    CHECK_CUDA(cudaMalloc(&model->d_patches, patches_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_patch_embeds, patch_embeds_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_time_embeds, time_embeds_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_input_embeds, input_embeds_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_noise_pred, patches_size * sizeof(float)));
    
    // Loss computation
    CHECK_CUDA(cudaMalloc(&model->d_loss_result, sizeof(float)));
    
    // Initialize curand states
    int total_curand_states = batch_size * NUM_PATCHES * PATCH_DIM;
    CHECK_CUDA(cudaMalloc(&model->d_curand_states, total_curand_states * sizeof(curandState)));
    
    int block_size = 256;
    int num_blocks = (total_curand_states + block_size - 1) / block_size;
    init_curand_states<<<num_blocks, block_size>>>(model->d_curand_states, time(NULL), total_curand_states);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Initialize embeddings on host then copy to device
    float* h_patch_projection = (float*)malloc(patch_proj_size * sizeof(float));
    float* h_time_embedding = (float*)malloc(time_emb_size * sizeof(float));
    
    float patch_scale = 1.0f / sqrtf(PATCH_DIM);
    float time_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < patch_proj_size; i++) {
        h_patch_projection[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * patch_scale;
    }
    
    for (int i = 0; i < time_emb_size; i++) {
        h_time_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * time_scale;
    }
    
    CHECK_CUDA(cudaMemcpy(model->d_patch_projection, h_patch_projection, patch_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(model->d_time_embedding, h_time_embedding, time_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(model->d_patch_projection_m, 0, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_patch_projection_v, 0, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_time_embedding_m, 0, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_time_embedding_v, 0, time_emb_size * sizeof(float)));
    
    // Initialize noise schedule on host
    model->h_betas = (float*)malloc(MAX_TIMESTEPS * sizeof(float));
    model->h_alphas = (float*)malloc(MAX_TIMESTEPS * sizeof(float));
    model->h_alpha_bars = (float*)malloc(MAX_TIMESTEPS * sizeof(float));
    
    for (int t = 0; t < MAX_TIMESTEPS; t++) {
        model->h_betas[t] = MIN_BETA + (MAX_BETA - MIN_BETA) * t / (MAX_TIMESTEPS - 1);
        model->h_alphas[t] = 1.0f - model->h_betas[t];
    }
    
    // Calculate cumulative alpha bars
    model->h_alpha_bars[0] = model->h_alphas[0];
    for (int t = 1; t < MAX_TIMESTEPS; t++) {
        model->h_alpha_bars[t] = model->h_alpha_bars[t-1] * model->h_alphas[t];
    }
    
    // Initialize transformer (non-causal for diffusion)
    model->transformer = init_transformer(model->seq_len, d_model, hidden_dim, num_layers, batch_size, false, cublaslt_handle);
    
    free(h_patch_projection);
    free(h_time_embedding);
    
    return model;
}

// Free diffusion model
void free_diffusion_model(DiffusionModel* model) {
    free_transformer(model->transformer);
    
    cudaFree(model->d_patch_projection); cudaFree(model->d_patch_projection_grad);
    cudaFree(model->d_time_embedding); cudaFree(model->d_time_embedding_grad);
    cudaFree(model->d_patch_projection_m); cudaFree(model->d_patch_projection_v);
    cudaFree(model->d_time_embedding_m); cudaFree(model->d_time_embedding_v);
    cudaFree(model->d_patches); cudaFree(model->d_patch_embeds);
    cudaFree(model->d_time_embeds); cudaFree(model->d_input_embeds);
    cudaFree(model->d_noise_pred); cudaFree(model->d_loss_result);
    cudaFree(model->d_curand_states);
    
    free(model->h_betas); free(model->h_alphas); free(model->h_alpha_bars);
    free(model);
}

// Convert images to patches
void image_to_patches(float* d_patches, float* d_images, int batch_size) {
    dim3 grid(batch_size, NUM_PATCHES);
    dim3 block(PATCH_DIM);
    image_to_patches_kernel<<<grid, block>>>(d_patches, d_images, batch_size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Convert patches to images
void patches_to_image(float* d_images, float* d_patches, int batch_size) {
    dim3 grid(batch_size, NUM_PATCHES);
    dim3 block(PATCH_DIM);
    patches_to_image_kernel<<<grid, block>>>(d_images, d_patches, batch_size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Forward diffusion process (add noise)
void forward_diffusion(DiffusionModel* model, float* d_noisy_patches, float* d_clean_patches, int* timesteps, int batch_size) {
    int total_elements = batch_size * NUM_PATCHES * PATCH_DIM;
    
    // Generate noise
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    generate_noise_kernel<<<num_blocks, block_size>>>(model->d_noise_pred, model->d_curand_states, total_elements);
    
    // Add noise based on timestep (simplified - using first timestep for all)
    int t = timesteps[0];  // Use first timestep
    float sqrt_alpha_bar = sqrtf(model->h_alpha_bars[t]);
    float sqrt_one_minus_alpha_bar = sqrtf(1.0f - model->h_alpha_bars[t]);
    
    add_noise_kernel<<<num_blocks, block_size>>>(d_noisy_patches, d_clean_patches, model->d_noise_pred,
                                                 sqrt_alpha_bar, sqrt_one_minus_alpha_bar, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Forward pass through diffusion model
void forward_pass_diffusion(DiffusionModel* model, float* d_patches, int* timesteps) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Project patches to embeddings
    CHECK_CUBLASLT(cublasLtMatmul(model->cublaslt_handle,
                                  model->transformer->attention_layers[0]->matmul_NN_desc,
                                  &alpha,
                                  d_patches, model->transformer->attention_layers[0]->flattened_seq_layout,
                                  model->d_patch_projection, model->transformer->attention_layers[0]->weight_layout,
                                  &beta,
                                  model->d_patch_embeds, model->transformer->attention_layers[0]->flattened_seq_layout,
                                  model->d_patch_embeds, model->transformer->attention_layers[0]->flattened_seq_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 2: Add time embeddings (broadcast first timestep to all batches)
    int t = timesteps[0];
    CHECK_CUDA(cudaMemcpy(model->d_time_embeds, &model->d_time_embedding[t * model->d_model], 
                         model->d_model * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 3: Combine patch and time embeddings
    // Copy patch embeddings to input
    CHECK_CUDA(cudaMemcpy(model->d_input_embeds, model->d_patch_embeds, 
                         model->batch_size * NUM_PATCHES * model->d_model * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Add time embedding as last token for each batch
    for (int b = 0; b < model->batch_size; b++) {
        int time_offset = b * model->seq_len * model->d_model + NUM_PATCHES * model->d_model;
        CHECK_CUDA(cudaMemcpy(&model->d_input_embeds[time_offset], model->d_time_embeds, 
                             model->d_model * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Step 4: Forward pass through transformer
    forward_pass_transformer(model->transformer, model->d_input_embeds);
    
    // Step 5: Project back to patch space (only patch tokens, not time token)
    float* transformer_output = model->transformer->mlp_layers[model->num_layers-1]->d_layer_output;
    CHECK_CUBLASLT(cublasLtMatmul(model->cublaslt_handle,
                                  model->transformer->attention_layers[0]->matmul_NT_desc,
                                  &alpha,
                                  transformer_output, model->transformer->attention_layers[0]->flattened_seq_layout,
                                  model->d_patch_projection, model->transformer->attention_layers[0]->weight_layout,
                                  &beta,
                                  model->d_noise_pred, model->transformer->attention_layers[0]->flattened_seq_layout,
                                  model->d_noise_pred, model->transformer->attention_layers[0]->flattened_seq_layout,
                                  NULL, NULL, 0, 0));
}

// Calculate diffusion loss (MSE between predicted and actual noise)
__global__ static void mse_loss_kernel(float* loss_result, float* pred_noise, float* actual_noise, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float diff = pred_noise[idx] - actual_noise[idx];
        atomicAdd(loss_result, diff * diff);
    }
}

float calculate_loss_diffusion(DiffusionModel* model, float* d_target_noise) {
    int total_elements = model->batch_size * NUM_PATCHES * PATCH_DIM;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    CHECK_CUDA(cudaMemset(model->d_loss_result, 0, sizeof(float)));
    
    mse_loss_kernel<<<num_blocks, block_size>>>(model->d_loss_result, model->d_noise_pred, d_target_noise, total_elements);
    
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, model->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_diffusion(DiffusionModel* model) {
    int patch_proj_size = PATCH_DIM * model->d_model;
    int time_emb_size = MAX_TIMESTEPS * model->d_model;
    
    CHECK_CUDA(cudaMemset(model->d_patch_projection_grad, 0, patch_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_time_embedding_grad, 0, time_emb_size * sizeof(float)));
    
    zero_gradients_transformer(model->transformer);
}

// Backward pass (simplified)
void backward_pass_diffusion(DiffusionModel* model, float* d_patches, int* timesteps) {
    // This is a simplified backward pass - in practice you'd need full backprop
    // For now, just call transformer backward pass
    backward_pass_transformer(model->transformer, model->d_input_embeds, model->d_input_embeds);
}

// Update weights
__global__ static void adamw_update_kernel_diffusion(float* weight, float* grad, float* m, float* v,
                                                     float beta1, float beta2, float epsilon, float learning_rate,
                                                     float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

void update_weights_diffusion(DiffusionModel* model, float learning_rate) {
    model->t++;
    
    float beta1_t = powf(model->beta1, model->t);
    float beta2_t = powf(model->beta2, model->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update patch projection
    int patch_proj_size = PATCH_DIM * model->d_model;
    int patch_blocks = (patch_proj_size + block_size - 1) / block_size;
    adamw_update_kernel_diffusion<<<patch_blocks, block_size>>>(
        model->d_patch_projection, model->d_patch_projection_grad, 
        model->d_patch_projection_m, model->d_patch_projection_v,
        model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay,
        alpha_t, patch_proj_size, model->batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(model->transformer, learning_rate);
}

// Convert CIFAR-10 images to float format
void cifar10_to_float(float* output, const CIFAR10_Image* images, int num_images) {
    for (int i = 0; i < num_images; i++) {
        for (int p = 0; p < CIFAR10_TOTAL_PIXELS; p++) {
            // Convert CIFAR-10 format (R[1024], G[1024], B[1024]) to interleaved RGB and normalize to [-1, 1]
            int pixel_idx = p % CIFAR10_PIXELS;
            int channel = p / CIFAR10_PIXELS;
            
            int rgb_idx = i * CIFAR10_TOTAL_PIXELS + pixel_idx * 3 + channel;
            output[rgb_idx] = (images[i].pixels[p] / 127.5f) - 1.0f;  // Normalize to [-1, 1]
        }
    }
}