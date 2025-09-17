#include "sim.h"

// Helper: initialize diffusion schedule on device arrays
static void init_diffusion_schedule(SIM* sim, int T) {
    sim->T = T;
    
    float* h_betas  = (float*)malloc(T * sizeof(float));
    float* h_alphas = (float*)malloc(T * sizeof(float));
    float* h_alphas_cumprod = (float*)malloc(T * sizeof(float));
    float* h_sqrt_alphas_cumprod = (float*)malloc(T * sizeof(float));
    float* h_sqrt_one_minus_alphas_cumprod = (float*)malloc(T * sizeof(float));
    float* h_sqrt_recip_alphas = (float*)malloc(T * sizeof(float));
    float* h_sqrt_betas = (float*)malloc(T * sizeof(float));
    
    // Linear beta schedule (DDPM) from beta_start to beta_end
    const float beta_start = 1e-4f;
    const float beta_end   = 2e-2f;
    for (int t = 0; t < T; t++) {
        h_betas[t] = beta_start + (beta_end - beta_start) * (float)t / (float)(T - 1);
    }
    // Compute alphas and alpha_bar
    float alpha_cumprod = 1.0f;
    for (int t = 0; t < T; t++) {
        float alpha = 1.0f - h_betas[t];
        h_alphas[t] = alpha;
        alpha_cumprod *= alpha;
        h_alphas_cumprod[t] = alpha_cumprod;
        
        h_sqrt_alphas_cumprod[t] = sqrtf(alpha_cumprod);
        h_sqrt_one_minus_alphas_cumprod[t] = sqrtf(1.0f - alpha_cumprod);
        h_sqrt_recip_alphas[t] = 1.0f / sqrtf(alpha);
        h_sqrt_betas[t] = sqrtf(h_betas[t]);
    }
    
    CHECK_CUDA(cudaMalloc(&sim->d_betas,  T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_alphas, T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_alphas_cumprod, T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_sqrt_alphas_cumprod, T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_sqrt_one_minus_alphas_cumprod, T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_sqrt_recip_alphas, T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_sqrt_betas, T * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(sim->d_betas,  h_betas,  T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_alphas, h_alphas, T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_alphas_cumprod, h_alphas_cumprod, T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_sqrt_alphas_cumprod, h_sqrt_alphas_cumprod, T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_sqrt_one_minus_alphas_cumprod, h_sqrt_one_minus_alphas_cumprod, T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_sqrt_recip_alphas, h_sqrt_recip_alphas, T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_sqrt_betas, h_sqrt_betas, T * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_betas);
    free(h_alphas);
    free(h_alphas_cumprod);
    free(h_sqrt_alphas_cumprod);
    free(h_sqrt_one_minus_alphas_cumprod);
    free(h_sqrt_recip_alphas);
    free(h_sqrt_betas);
}

// CUDA kernel: scalar -> embedding (per token)
// e[b,t,d] = x_t[b,t] * w_in[d] + b_in[d]
__global__ static void scalar_to_embedding_forward_kernel(float* embedded, const float* x_t, const float* w_in, const float* b_in,
                                                          int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int emb_idx = (b * seq_len + t) * d_model + d;
    float x = x_t[b * seq_len + t];
    embedded[emb_idx] = x * w_in[d] + b_in[d];
}

// CUDA kernel: add 2D positional + class + time encoding
// Split d_model into: pos_dims = d_model/2, class_dims = d_model/4, time_dims = remaining
__global__ static void add_pos_class_time_encoding_kernel(float* embedded,
                                                          const unsigned char* class_labels,
                                                          int batch_size, int seq_len, int d_model,
                                                          int image_size, int num_classes,
                                                          int t_index, int T_total) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int idx = (b * seq_len + t) * d_model + d;
    
    // 2D position
    int row = t / image_size;
    int col = t % image_size;
    int class_label = class_labels[b];
    
    int pos_dims = d_model / 2;
    int rem = d_model - pos_dims;
    int class_dims = rem / 2;
    int time_dims  = d_model - pos_dims - class_dims;
    
    float add = 0.0f;
    
    // Positional encoding (row/col, sinusoidal)
    if (d < pos_dims) {
        int half = pos_dims / 2;
        if (d < half) {
            // Row
            int k = d;
            float denom = powf(10000.0f, (2.0f * (k / 2)) / (float)(half));
            if ((k % 2) == 0) add = sinf(row / denom);
            else               add = cosf(row / denom);
        } else {
            // Col
            int k = d - half;
            float denom = powf(10000.0f, (2.0f * (k / 2)) / (float)(half));
            if ((k % 2) == 0) add = sinf(col / denom);
            else               add = cosf(col / denom);
        }
    }
    // Class encoding (sinusoid over discrete class)
    else if (d < pos_dims + class_dims) {
        int k = d - pos_dims;
        float denom = powf(10000.0f, (2.0f * (k / 2)) / (float)class_dims);
        if ((k % 2) == 0) add = sinf(class_label / denom);
        else               add = cosf(class_label / denom);
    }
    // Time encoding (t scaled to [0,1])
    else {
        int k = d - pos_dims - class_dims;
        float t_scaled = (float)t_index / (float)(T_total - 1);
        float denom = powf(10000.0f, (2.0f * (k / 2)) / (float)time_dims);
        if ((k % 2) == 0) add = sinf(t_scaled / denom);
        else               add = cosf(t_scaled / denom);
    }
    
    embedded[idx] += add;
}

// CUDA kernel: compute gradients for input projection w_in and b_in
// dL/dw_in[d] = sum_{b,t} grad_embedded[b,t,d] * x_t[b,t]
// dL/db_in[d] = sum_{b,t} grad_embedded[b,t,d]
__global__ static void in_proj_backward_kernel(float* w_grad, float* b_grad,
                                               const float* grad_embedded, const float* x_t,
                                               int batch_size, int seq_len, int d_model) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_model) return;
    
    float sum_w = 0.0f;
    float sum_b = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int emb_idx = (b * seq_len + t) * d_model + d;
            float ge = grad_embedded[emb_idx];
            sum_w += ge * x_t[b * seq_len + t];
            sum_b += ge;
        }
    }
    w_grad[d] += sum_w;
    b_grad[d] += sum_b;
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_sim(float* weight, float* grad, float* m, float* v,
                                               float beta1, float beta2, float epsilon, float learning_rate,
                                               float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size; // normalize by batch_size
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Initialize the SIM (Diffusion model)
SIM* init_sim(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, int T, cublasLtHandle_t cublaslt_handle) {
    SIM* sim = (SIM*)malloc(sizeof(SIM));
    
    // Store dimensions and handle
    sim->seq_len = seq_len;
    sim->d_model = d_model;
    sim->batch_size = batch_size;
    sim->hidden_dim = hidden_dim;
    sim->num_layers = num_layers;
    sim->image_size = 28; // MNIST images 28x28
    sim->num_classes = 10;
    sim->cublaslt_handle = cublaslt_handle;
    
    // AdamW hyperparams
    sim->beta1 = 0.9f;
    sim->beta2 = 0.999f;
    sim->epsilon = 1e-8f;
    sim->t = 0;
    sim->weight_decay = 0.01f;
    
    // Initialize diffusion schedule
    init_diffusion_schedule(sim, T);
    
    // Allocate input projection weights and related buffers
    int D = d_model;
    float* h_w = (float*)malloc(D * sizeof(float));
    float* h_b = (float*)malloc(D * sizeof(float));
    float scale = 1.0f / sqrtf((float)D);
    for (int i = 0; i < D; i++) {
        h_w[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        h_b[i] = 0.0f;
    }
    
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_w, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_b, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_w_grad, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_b_grad, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_w_m, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_w_v, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_b_m, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_in_proj_b_v, D * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_w, h_w, D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_w_grad, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_b_grad, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_w_m, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_w_v, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_b_m, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_b_v, 0, D * sizeof(float)));
    
    free(h_w); free(h_b);
    
    // Allocate embedded buffer
    CHECK_CUDA(cudaMalloc(&sim->d_embedded_input, batch_size * seq_len * d_model * sizeof(float)));
    
    // Initialize transformer (causal=false; diffusion uses bidirectional context)
    sim->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, false, cublaslt_handle);
    
    // Initialize output MLP: d_model -> 1 (epsilon per pixel)
    sim->output_mlp = init_mlp(d_model, hidden_dim, 1, batch_size * seq_len, cublaslt_handle);
    
    return sim;
}

// Free SIM memory
void free_sim(SIM* sim) {
    // Destroy submodules
    free_transformer(sim->transformer);
    free_mlp(sim->output_mlp);
    
    // Free input projection
    cudaFree(sim->d_in_proj_w); cudaFree(sim->d_in_proj_b);
    cudaFree(sim->d_in_proj_w_grad); cudaFree(sim->d_in_proj_b_grad);
    cudaFree(sim->d_in_proj_w_m); cudaFree(sim->d_in_proj_w_v);
    cudaFree(sim->d_in_proj_b_m); cudaFree(sim->d_in_proj_b_v);
    
    // Free buffers
    cudaFree(sim->d_embedded_input);
    
    // Free diffusion schedule
    cudaFree(sim->d_betas);
    cudaFree(sim->d_alphas);
    cudaFree(sim->d_alphas_cumprod);
    cudaFree(sim->d_sqrt_alphas_cumprod);
    cudaFree(sim->d_sqrt_one_minus_alphas_cumprod);
    cudaFree(sim->d_sqrt_recip_alphas);
    cudaFree(sim->d_sqrt_betas);
    
    free(sim);
}

// Forward pass
// d_x_t: [batch_size * seq_len] noised images at time t_index (1..T)
// d_class_labels: [batch_size]
// Adds positional+class+time encodings and passes through transformer + head
void forward_pass_sim(SIM* sim, float* d_x_t, unsigned char* d_class_labels, int t_index) {
    dim3 grid(sim->batch_size, sim->seq_len);
    dim3 block(sim->d_model);
    
    // 1) Scalar -> embedding
    scalar_to_embedding_forward_kernel<<<grid, block>>>(
        sim->d_embedded_input, d_x_t, sim->d_in_proj_w, sim->d_in_proj_b,
        sim->batch_size, sim->seq_len, sim->d_model
    );
    
    // 2) Add 2D position + class + time encoding
    add_pos_class_time_encoding_kernel<<<grid, block>>>(
        sim->d_embedded_input, d_class_labels,
        sim->batch_size, sim->seq_len, sim->d_model,
        sim->image_size, sim->num_classes,
        t_index, sim->T
    );
    
    // 3) Transformer
    forward_pass_transformer(sim->transformer, sim->d_embedded_input);
    
    // 4) Output head (predict epsilon per pixel)
    forward_pass_mlp(sim->output_mlp, sim->transformer->mlp_layers[sim->num_layers-1]->d_layer_output);
}

// Calculate MSE loss between predicted epsilon and target epsilon
float calculate_loss_sim(SIM* sim, float* d_noise_target) {
    return calculate_loss_mlp(sim->output_mlp, d_noise_target);
}

// Zero gradients
void zero_gradients_sim(SIM* sim) {
    int D = sim->d_model;
    CHECK_CUDA(cudaMemset(sim->d_in_proj_w_grad, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_in_proj_b_grad, 0, D * sizeof(float)));
    
    zero_gradients_transformer(sim->transformer);
    zero_gradients_mlp(sim->output_mlp);
}

// Backward pass
// Computes grads for output head -> transformer -> input projection
void backward_pass_sim(SIM* sim, float* d_x_t) {
    // Backprop through head: gives grad into transformer last layer output
    backward_pass_mlp(sim->output_mlp,
                      sim->transformer->mlp_layers[sim->num_layers-1]->d_layer_output,
                      sim->transformer->mlp_layers[sim->num_layers-1]->d_grad_output);
    
    // Backprop through transformer: gives grad into sim->d_embedded_input
    backward_pass_transformer(sim->transformer, sim->d_embedded_input, sim->d_embedded_input);
    
    // Compute grads for input projection
    int block = 256;
    int grid = (sim->d_model + block - 1) / block;
    in_proj_backward_kernel<<<grid, block>>>(
        sim->d_in_proj_w_grad, sim->d_in_proj_b_grad,
        sim->d_embedded_input, d_x_t,
        sim->batch_size, sim->seq_len, sim->d_model
    );
}

// Update weights
void update_weights_sim(SIM* sim, float learning_rate) {
    sim->t++;
    float beta1_t = powf(sim->beta1, sim->t);
    float beta2_t = powf(sim->beta2, sim->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int D = sim->d_model;
    int block = 256;
    int grid = (D + block - 1) / block;
    
    // Update input projection weights and biases
    adamw_update_kernel_sim<<<grid, block>>>(
        sim->d_in_proj_w, sim->d_in_proj_w_grad, sim->d_in_proj_w_m, sim->d_in_proj_w_v,
        sim->beta1, sim->beta2, sim->epsilon, learning_rate, sim->weight_decay,
        alpha_t, D, sim->batch_size
    );
    adamw_update_kernel_sim<<<grid, block>>>(
        sim->d_in_proj_b, sim->d_in_proj_b_grad, sim->d_in_proj_b_m, sim->d_in_proj_b_v,
        sim->beta1, sim->beta2, sim->epsilon, learning_rate, sim->weight_decay,
        alpha_t, D, sim->batch_size
    );
    
    // Update submodules
    update_weights_transformer(sim->transformer, learning_rate);
    update_weights_mlp(sim->output_mlp, learning_rate);
}

// Save SIM to binary (plus submodules to their files)
// We save: dims, T, input projection weights/bias, Adam states, and then
// save transformer/output_mlp to side files with base name + suffix.
void save_sim(SIM* sim, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions + T
    fwrite(&sim->seq_len, sizeof(int), 1, file);
    fwrite(&sim->d_model, sizeof(int), 1, file);
    fwrite(&sim->batch_size, sizeof(int), 1, file);
    fwrite(&sim->hidden_dim, sizeof(int), 1, file);
    fwrite(&sim->num_layers, sizeof(int), 1, file);
    fwrite(&sim->image_size, sizeof(int), 1, file);
    fwrite(&sim->num_classes, sizeof(int), 1, file);
    fwrite(&sim->T, sizeof(int), 1, file);
    
    int D = sim->d_model;
    float* h_w = (float*)malloc(D * sizeof(float));
    float* h_b = (float*)malloc(D * sizeof(float));
    float* h_w_m = (float*)malloc(D * sizeof(float));
    float* h_w_v = (float*)malloc(D * sizeof(float));
    float* h_b_m = (float*)malloc(D * sizeof(float));
    float* h_b_v = (float*)malloc(D * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_w, sim->d_in_proj_w, D * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, sim->d_in_proj_b, D * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_w, sizeof(float), D, file);
    fwrite(h_b, sizeof(float), D, file);
    
    // Save Adam state
    fwrite(&sim->t, sizeof(int), 1, file);
    CHECK_CUDA(cudaMemcpy(h_w_m, sim->d_in_proj_w_m, D * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_w_v, sim->d_in_proj_w_v, D * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_m, sim->d_in_proj_b_m, D * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_v, sim->d_in_proj_b_v, D * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_w_m, sizeof(float), D, file);
    fwrite(h_w_v, sizeof(float), D, file);
    fwrite(h_b_m, sizeof(float), D, file);
    fwrite(h_b_v, sizeof(float), D, file);
    
    fclose(file);
    
    // Save transformer and output head with base filename
    char base_filename[256];
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    char transformer_filename[256];
    char output_mlp_filename[256];
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    snprintf(output_mlp_filename, sizeof(output_mlp_filename), "%s_output_mlp.bin", base_filename);
    
    save_transformer(sim->transformer, transformer_filename);
    save_mlp(sim->output_mlp, output_mlp_filename);
    
    free(h_w); free(h_b);
    free(h_w_m); free(h_w_v); free(h_b_m); free(h_b_v);
    
    printf("Model saved to %s\n", filename);
}

// Load SIM from binary file
SIM* load_sim(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int seq_len, d_model, stored_batch_size, hidden_dim, num_layers, image_size, num_classes, T;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&image_size, sizeof(int), 1, file);
    fread(&num_classes, sizeof(int), 1, file);
    fread(&T, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    SIM* sim = init_sim(seq_len, d_model, hidden_dim, num_layers, batch_size, T, cublaslt_handle);
    sim->image_size = image_size;
    sim->num_classes = num_classes;
    
    int D = d_model;
    float* h_w = (float*)malloc(D * sizeof(float));
    float* h_b = (float*)malloc(D * sizeof(float));
    fread(h_w, sizeof(float), D, file);
    fread(h_b, sizeof(float), D, file);
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_w, h_w, D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(&sim->t, sizeof(int), 1, file);
    float* h_w_m = (float*)malloc(D * sizeof(float));
    float* h_w_v = (float*)malloc(D * sizeof(float));
    float* h_b_m = (float*)malloc(D * sizeof(float));
    float* h_b_v = (float*)malloc(D * sizeof(float));
    fread(h_w_m, sizeof(float), D, file);
    fread(h_w_v, sizeof(float), D, file);
    fread(h_b_m, sizeof(float), D, file);
    fread(h_b_v, sizeof(float), D, file);
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_w_m, h_w_m, D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_w_v, h_w_v, D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_b_m, h_b_m, D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_in_proj_b_v, h_b_v, D * sizeof(float), cudaMemcpyHostToDevice));
    
    fclose(file);
    
    // Load submodules
    char base_filename[256];
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    char transformer_filename[256];
    char output_mlp_filename[256];
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    snprintf(output_mlp_filename, sizeof(output_mlp_filename), "%s_output_mlp.bin", base_filename);
    
    // Replace initialized submodules with loaded ones
    free_transformer(sim->transformer);
    free_mlp(sim->output_mlp);
    sim->transformer = load_transformer(transformer_filename, batch_size, cublaslt_handle);
    sim->output_mlp = load_mlp(output_mlp_filename, batch_size * seq_len, cublaslt_handle);
    
    free(h_w); free(h_b);
    free(h_w_m); free(h_w_v); free(h_b_m); free(h_b_v);
    
    printf("Model loaded from %s\n", filename);
    return sim;
}