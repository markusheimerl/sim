#include "sim.h"

// Initialize curand states
__global__ static void init_curand_states(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Convert RGB images to patches
__global__ static void image_to_patches_kernel(float* patches, float* images, int batch_size) {
    int b = blockIdx.x;
    int patch_idx = blockIdx.y;
    int pixel_in_patch = threadIdx.x;
    
    if (b >= batch_size || patch_idx >= NUM_PATCHES || pixel_in_patch >= PATCH_DIM) return;
    
    // Calculate patch position
    int patch_row = patch_idx / PATCHES_PER_ROW;
    int patch_col = patch_idx % PATCHES_PER_ROW;
    
    // Calculate pixel position within patch
    int pixel_idx = pixel_in_patch / 3;
    int channel = pixel_in_patch % 3;
    
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

// Generate random noise using curand
__global__ static void generate_noise_kernel(float* noise, curandState* states, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        noise[idx] = curand_normal(&states[idx]);
    }
}

// Add noise to clean patches (forward diffusion process)
__global__ static void add_noise_kernel(float* noisy_patches, float* clean_patches, float* noise,
                                       float sqrt_alpha_bar, float sqrt_one_minus_alpha_bar,
                                       int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        noisy_patches[idx] = sqrt_alpha_bar * clean_patches[idx] + sqrt_one_minus_alpha_bar * noise[idx];
    }
}

// Sinusoidal position encoding addition (for patches)
__global__ static void sinusoidal_position_encoding_kernel(float* embedded, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int idx = b * seq_len * d_model + t * d_model + d;
    float pos_encoding;
    
    if (d % 2 == 0) {
        pos_encoding = sinf(t / powf(10000.0f, (2.0f * (d / 2)) / d_model));
    } else {
        pos_encoding = cosf(t / powf(10000.0f, (2.0f * ((d - 1) / 2)) / d_model));
    }
    
    embedded[idx] += pos_encoding;
}

// Add time embedding to all patch embeddings
__global__ static void add_time_embedding_kernel(float* patch_embeds, float* time_embedding, 
                                                int timestep, int batch_size, int d_model) {
    int b = blockIdx.x;
    int p = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || p >= NUM_PATCHES || d >= d_model) return;
    
    int embed_idx = b * NUM_PATCHES * d_model + p * d_model + d;
    patch_embeds[embed_idx] += time_embedding[timestep * d_model + d];
}

// MSE loss kernel for noise prediction
__global__ static void mse_loss_kernel(float* loss_result, float* pred_noise, float* actual_noise, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float diff = pred_noise[idx] - actual_noise[idx];
        atomicAdd(loss_result, diff * diff);
    }
}

// Compute MSE gradients (standard implementation)
__global__ static void mse_gradients_kernel(float* grad_output, float* pred_noise, float* actual_noise, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        grad_output[idx] = pred_noise[idx] - actual_noise[idx];
    }
}

// Time embedding gradient accumulation
__global__ static void time_embedding_grad_kernel(float* time_embedding_grad, float* patch_embed_grads,
                                                  int timestep, int batch_size, int d_model) {
    int b = blockIdx.x;
    int p = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || p >= NUM_PATCHES || d >= d_model) return;
    
    int embed_idx = b * NUM_PATCHES * d_model + p * d_model + d;
    atomicAdd(&time_embedding_grad[timestep * d_model + d], patch_embed_grads[embed_idx]);
}

// Initialize the SIM
SIM* init_sim(int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    SIM* sim = (SIM*)malloc(sizeof(SIM));
    
    // Store dimensions
    sim->d_model = d_model;
    sim->batch_size = batch_size;
    sim->hidden_dim = hidden_dim;
    sim->num_layers = num_layers;
    sim->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    sim->beta1 = 0.9f;
    sim->beta2 = 0.999f;
    sim->epsilon = 1e-8f;
    sim->t = 0;
    sim->weight_decay = 0.01f;
    
    int time_emb_size = MAX_TIMESTEPS * d_model;
    int patch_buffer_size = batch_size * NUM_PATCHES * PATCH_DIM;
    int patch_embed_size = batch_size * NUM_PATCHES * d_model;
    int curand_states_size = patch_buffer_size;
    
    // Allocate host memory for time embedding initialization
    float* h_time_embedding = (float*)malloc(time_emb_size * sizeof(float));
    
    // Initialize time embeddings on host
    float time_scale = 1.0f / sqrtf(d_model);
    for (int i = 0; i < time_emb_size; i++) {
        h_time_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * time_scale;
    }
    
    // Allocate device memory for time embeddings
    CHECK_CUDA(cudaMalloc(&sim->d_time_embedding, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_time_embedding_grad, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_time_embedding_m, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_time_embedding_v, time_emb_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&sim->d_patches, patch_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_noisy_patches, patch_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_noise, patch_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_patch_embeds, patch_embed_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_loss_result, sizeof(float)));
    
    // Initialize curand states
    CHECK_CUDA(cudaMalloc(&sim->d_curand_states, curand_states_size * sizeof(curandState)));
    
    int block_size = 256;
    int num_blocks = (curand_states_size + block_size - 1) / block_size;
    init_curand_states<<<num_blocks, block_size>>>(sim->d_curand_states, time(NULL), curand_states_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy time embeddings to device
    CHECK_CUDA(cudaMemcpy(sim->d_time_embedding, h_time_embedding, time_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(sim->d_time_embedding_m, 0, time_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_time_embedding_v, 0, time_emb_size * sizeof(float)));
    
    // Initialize noise schedule on host
    sim->h_betas = (float*)malloc(MAX_TIMESTEPS * sizeof(float));
    sim->h_alphas = (float*)malloc(MAX_TIMESTEPS * sizeof(float));
    sim->h_alpha_bars = (float*)malloc(MAX_TIMESTEPS * sizeof(float));
    
    for (int t = 0; t < MAX_TIMESTEPS; t++) {
        sim->h_betas[t] = MIN_BETA + (MAX_BETA - MIN_BETA) * t / (MAX_TIMESTEPS - 1);
        sim->h_alphas[t] = 1.0f - sim->h_betas[t];
    }
    
    sim->h_alpha_bars[0] = sim->h_alphas[0];
    for (int t = 1; t < MAX_TIMESTEPS; t++) {
        sim->h_alpha_bars[t] = sim->h_alpha_bars[t-1] * sim->h_alphas[t];
    }
    
    // Initialize transformer (non-causal for diffusion, operates on NUM_PATCHES)
    sim->transformer = init_transformer(NUM_PATCHES, d_model, hidden_dim, num_layers, batch_size, false, cublaslt_handle);
    
    // Initialize input projection MLP (PATCH_DIM -> d_model)
    sim->input_mlp = init_mlp(PATCH_DIM, hidden_dim, d_model, batch_size * NUM_PATCHES, cublaslt_handle);
    
    // Initialize output projection MLP (d_model -> PATCH_DIM)
    sim->output_mlp = init_mlp(d_model, hidden_dim, PATCH_DIM, batch_size * NUM_PATCHES, cublaslt_handle);
    
    // Free host memory
    free(h_time_embedding);
    
    return sim;
}

// Free SIM memory
void free_sim(SIM* sim) {
    // Free transformer
    free_transformer(sim->transformer);
    
    // Free MLPs
    free_mlp(sim->input_mlp);
    free_mlp(sim->output_mlp);
    
    // Free device memory
    cudaFree(sim->d_time_embedding); cudaFree(sim->d_time_embedding_grad);
    cudaFree(sim->d_time_embedding_m); cudaFree(sim->d_time_embedding_v);
    cudaFree(sim->d_patches); cudaFree(sim->d_noisy_patches); cudaFree(sim->d_noise);
    cudaFree(sim->d_patch_embeds); cudaFree(sim->d_loss_result); cudaFree(sim->d_curand_states);
    
    // Free host memory
    free(sim->h_betas); free(sim->h_alphas); free(sim->h_alpha_bars);
    
    free(sim);
}

// Forward pass
void forward_pass_sim(SIM* sim, float* d_clean_images, int timestep) {
    int patch_buffer_size = sim->batch_size * NUM_PATCHES * PATCH_DIM;
    
    // Step 1: Convert images to patches
    dim3 grid_patches(sim->batch_size, NUM_PATCHES);
    dim3 block_patches(PATCH_DIM);
    image_to_patches_kernel<<<grid_patches, block_patches>>>(sim->d_patches, d_clean_images, sim->batch_size);
    
    // Step 2: Generate noise
    int block_size = 256;
    int num_blocks = (patch_buffer_size + block_size - 1) / block_size;
    generate_noise_kernel<<<num_blocks, block_size>>>(sim->d_noise, sim->d_curand_states, patch_buffer_size);
    
    // Step 3: Add noise (forward diffusion)
    float sqrt_alpha_bar = sqrtf(sim->h_alpha_bars[timestep]);
    float sqrt_one_minus_alpha_bar = sqrtf(1.0f - sim->h_alpha_bars[timestep]);
    add_noise_kernel<<<num_blocks, block_size>>>(sim->d_noisy_patches, sim->d_patches, sim->d_noise,
                                                 sqrt_alpha_bar, sqrt_one_minus_alpha_bar, patch_buffer_size);
    
    // Step 4: Project noisy patches to d_model dimension
    forward_pass_mlp(sim->input_mlp, sim->d_noisy_patches);
    CHECK_CUDA(cudaMemcpy(sim->d_patch_embeds, sim->input_mlp->d_layer_output, 
                         sim->batch_size * NUM_PATCHES * sim->d_model * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 5: Add position encodings to patch embeddings
    dim3 grid_pos(sim->batch_size, NUM_PATCHES);
    dim3 block_pos(sim->d_model);
    sinusoidal_position_encoding_kernel<<<grid_pos, block_pos>>>(sim->d_patch_embeds, sim->batch_size, NUM_PATCHES, sim->d_model);
    
    // Step 6: Add time embeddings to all patch embeddings
    dim3 grid_time(sim->batch_size, NUM_PATCHES);
    dim3 block_time(sim->d_model);
    add_time_embedding_kernel<<<grid_time, block_time>>>(sim->d_patch_embeds, sim->d_time_embedding, 
                                                         timestep, sim->batch_size, sim->d_model);
    
    // Step 7: Forward pass through transformer
    forward_pass_transformer(sim->transformer, sim->d_patch_embeds);
    
    // Step 8: Project back to patch dimension (noise prediction)
    forward_pass_mlp(sim->output_mlp, sim->transformer->mlp_layers[sim->num_layers-1]->d_layer_output);
}

// Calculate loss
float calculate_loss_sim(SIM* sim) {
    int total_elements = sim->batch_size * NUM_PATCHES * PATCH_DIM;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    CHECK_CUDA(cudaMemset(sim->d_loss_result, 0, sizeof(float)));
    
    // MSE loss between predicted noise and actual noise
    mse_loss_kernel<<<num_blocks, block_size>>>(sim->d_loss_result, sim->output_mlp->d_layer_output, sim->d_noise, total_elements);
    
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, sim->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_sim(SIM* sim) {
    int time_emb_size = MAX_TIMESTEPS * sim->d_model;
    
    CHECK_CUDA(cudaMemset(sim->d_time_embedding_grad, 0, time_emb_size * sizeof(float)));
    
    zero_gradients_transformer(sim->transformer);
    zero_gradients_mlp(sim->input_mlp);
    zero_gradients_mlp(sim->output_mlp);
}

// Backward pass
void backward_pass_sim(SIM* sim, int timestep) {
    int total_elements = sim->batch_size * NUM_PATCHES * PATCH_DIM;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Step 8 (backward): Compute MSE gradients for output MLP
    mse_gradients_kernel<<<num_blocks, block_size>>>(sim->output_mlp->d_grad_output, 
                                                     sim->output_mlp->d_layer_output, sim->d_noise, total_elements);
    
    // Step 7 (backward): Backward through output MLP
    backward_pass_mlp(sim->output_mlp, sim->transformer->mlp_layers[sim->num_layers-1]->d_layer_output, 
                      sim->transformer->mlp_layers[sim->num_layers-1]->d_grad_output);
    
    // Step 6 (backward): Backward through transformer
    backward_pass_transformer(sim->transformer, sim->d_patch_embeds, sim->d_patch_embeds);
    
    // Step 5 (backward): Accumulate time embedding gradients
    dim3 grid_time_grad(sim->batch_size, NUM_PATCHES);
    dim3 block_time_grad(sim->d_model);
    time_embedding_grad_kernel<<<grid_time_grad, block_time_grad>>>(sim->d_time_embedding_grad, sim->d_patch_embeds,
                                                                    timestep, sim->batch_size, sim->d_model);
    
    // Step 4 (backward): Position encoding gradients pass through unchanged
    
    // Step 3 (backward): Backward through input MLP
    backward_pass_mlp(sim->input_mlp, sim->d_noisy_patches, NULL);
    
    // Steps 1-2: No gradients needed for noise generation and image-to-patches conversion
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_sim(float* weight, float* grad, float* m, float* v,
                                               float beta1, float beta2, float epsilon, float learning_rate,
                                               float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights
void update_weights_sim(SIM* sim, float learning_rate) {
    sim->t++;
    
    float beta1_t = powf(sim->beta1, sim->t);
    float beta2_t = powf(sim->beta2, sim->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update time embeddings
    int time_emb_size = MAX_TIMESTEPS * sim->d_model;
    int time_blocks = (time_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_sim<<<time_blocks, block_size>>>(
        sim->d_time_embedding, sim->d_time_embedding_grad, sim->d_time_embedding_m, sim->d_time_embedding_v,
        sim->beta1, sim->beta2, sim->epsilon, learning_rate, sim->weight_decay,
        alpha_t, time_emb_size, sim->batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(sim->transformer, learning_rate);
    
    // Update MLP weights
    update_weights_mlp(sim->input_mlp, learning_rate);
    update_weights_mlp(sim->output_mlp, learning_rate);
}

// Save SIM to binary file
void save_sim(SIM* sim, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&sim->d_model, sizeof(int), 1, file);
    fwrite(&sim->batch_size, sizeof(int), 1, file);
    fwrite(&sim->hidden_dim, sizeof(int), 1, file);
    fwrite(&sim->num_layers, sizeof(int), 1, file);
    
    int time_emb_size = MAX_TIMESTEPS * sim->d_model;
    
    // Allocate host memory and copy time embeddings
    float* h_time_embedding = (float*)malloc(time_emb_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_time_embedding, sim->d_time_embedding, time_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_time_embedding, sizeof(float), time_emb_size, file);
    
    // Save Adam state
    fwrite(&sim->t, sizeof(int), 1, file);
    
    float* h_time_embedding_m = (float*)malloc(time_emb_size * sizeof(float));
    float* h_time_embedding_v = (float*)malloc(time_emb_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_time_embedding_m, sim->d_time_embedding_m, time_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_time_embedding_v, sim->d_time_embedding_v, time_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_time_embedding_m, sizeof(float), time_emb_size, file);
    fwrite(h_time_embedding_v, sizeof(float), time_emb_size, file);
    
    fclose(file);
    
    // Save transformer and MLP components
    char transformer_filename[256];
    char input_mlp_filename[256];
    char output_mlp_filename[256];
    char base_filename[256];
    
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    snprintf(input_mlp_filename, sizeof(input_mlp_filename), "%s_input_mlp.bin", base_filename);
    snprintf(output_mlp_filename, sizeof(output_mlp_filename), "%s_output_mlp.bin", base_filename);
    
    save_transformer(sim->transformer, transformer_filename);
    save_mlp(sim->input_mlp, input_mlp_filename);
    save_mlp(sim->output_mlp, output_mlp_filename);
    
    free(h_time_embedding);
    free(h_time_embedding_m);
    free(h_time_embedding_v);
    
    printf("Model saved to %s\n", filename);
}

// Load SIM from binary file
SIM* load_sim(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int d_model, stored_batch_size, hidden_dim, num_layers;
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize SIM
    SIM* sim = init_sim(d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    int time_emb_size = MAX_TIMESTEPS * d_model;
    
    // Load time embeddings
    float* h_time_embedding = (float*)malloc(time_emb_size * sizeof(float));
    fread(h_time_embedding, sizeof(float), time_emb_size, file);
    CHECK_CUDA(cudaMemcpy(sim->d_time_embedding, h_time_embedding, time_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&sim->t, sizeof(int), 1, file);
    
    float* h_time_embedding_m = (float*)malloc(time_emb_size * sizeof(float));
    float* h_time_embedding_v = (float*)malloc(time_emb_size * sizeof(float));
    
    fread(h_time_embedding_m, sizeof(float), time_emb_size, file);
    fread(h_time_embedding_v, sizeof(float), time_emb_size, file);
    
    CHECK_CUDA(cudaMemcpy(sim->d_time_embedding_m, h_time_embedding_m, time_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_time_embedding_v, h_time_embedding_v, time_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fclose(file);
    
    // Load transformer and MLP components
    char transformer_filename[256];
    char input_mlp_filename[256];
    char output_mlp_filename[256];
    char base_filename[256];
    
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    snprintf(input_mlp_filename, sizeof(input_mlp_filename), "%s_input_mlp.bin", base_filename);
    snprintf(output_mlp_filename, sizeof(output_mlp_filename), "%s_output_mlp.bin", base_filename);
    
    // Free initialized components
    free_transformer(sim->transformer);
    free_mlp(sim->input_mlp);
    free_mlp(sim->output_mlp);
    
    // Load saved components
    sim->transformer = load_transformer(transformer_filename, batch_size, cublaslt_handle);
    sim->input_mlp = load_mlp(input_mlp_filename, batch_size * NUM_PATCHES, cublaslt_handle);
    sim->output_mlp = load_mlp(output_mlp_filename, batch_size * NUM_PATCHES, cublaslt_handle);
    
    free(h_time_embedding);
    free(h_time_embedding_m);
    free(h_time_embedding_v);
    
    printf("Model loaded from %s\n", filename);
    return sim;
}