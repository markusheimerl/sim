#include "sim.h"

// Initialize the SIM
SIM* init_sim(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    SIM* sim = (SIM*)malloc(sizeof(SIM));
    
    // Store dimensions
    sim->seq_len = seq_len;
    sim->d_model = d_model;
    sim->batch_size = batch_size;
    sim->hidden_dim = hidden_dim;
    sim->num_layers = num_layers;
    sim->vocab_size = 256;
    sim->image_size = 28; // MNIST images are 28x28
    sim->num_classes = 10; // MNIST digits 0-9
    sim->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    sim->beta1 = 0.9f;
    sim->beta2 = 0.999f;
    sim->epsilon = 1e-8f;
    sim->t = 0;
    sim->weight_decay = 0.01f;
    
    int token_emb_size = sim->vocab_size * d_model;
    int embedded_size = batch_size * seq_len * d_model;
    
    // Allocate host memory for embedding initialization
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    
    // Initialize token embeddings on host
    float token_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < token_emb_size; i++) {
        h_token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Allocate device memory for embeddings and gradients
    CHECK_CUDA(cudaMalloc(&sim->d_token_embedding, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_token_embedding_grad, token_emb_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&sim->d_token_embedding_m, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sim->d_token_embedding_v, token_emb_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&sim->d_embedded_input, embedded_size * sizeof(float)));
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&sim->d_loss_result, sizeof(float)));
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(sim->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(sim->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(sim->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    
    // Initialize transformer
    sim->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, cublaslt_handle);
    
    // Initialize output projection MLP
    sim->output_mlp = init_mlp(d_model, hidden_dim, sim->vocab_size, batch_size * seq_len, cublaslt_handle);
    
    // Free host memory
    free(h_token_embedding);
    
    return sim;
}

// Free SIM memory
void free_sim(SIM* sim) {
    // Free transformer
    free_transformer(sim->transformer);
    
    // Free output MLP
    free_mlp(sim->output_mlp);
    
    // Free device memory
    cudaFree(sim->d_token_embedding); cudaFree(sim->d_token_embedding_grad);
    cudaFree(sim->d_token_embedding_m); cudaFree(sim->d_token_embedding_v);
    cudaFree(sim->d_embedded_input);
    
    // Free loss computation buffer
    cudaFree(sim->d_loss_result);
    
    free(sim);
}

// CUDA kernel for token embedding lookup
__global__ static void token_embedding_lookup_kernel(float* embedded, float* token_embedding, unsigned char* tokens, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int token_idx = b * seq_len + t;
    int token = tokens[token_idx];
    int emb_idx = b * seq_len * d_model + t * d_model + d;
    
    embedded[emb_idx] = token_embedding[token * d_model + d];
}

// CUDA kernel for 2D positional encodings and class label conditioning
__global__ static void positional_and_class_encoding_kernel(float* embedded, unsigned char* class_labels, int batch_size, int seq_len, int d_model, int image_size) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int idx = b * seq_len * d_model + t * d_model + d;
    
    // Convert 1D position to 2D coordinates
    int row = t / image_size;
    int col = t % image_size;
    
    // Get class label for this batch element
    int class_label = class_labels[b];
    
    float encoding = 0.0f;
    
    // Split d_model into three parts: position (2/3), class (1/3)
    int pos_dims = (2 * d_model) / 3;
    int class_dims = d_model - pos_dims;
    
    if (d < pos_dims / 2) {
        // First part: row encoding
        if (d % 2 == 0) {
            encoding = sinf(row / powf(10000.0f, (2.0f * (d / 2)) / (pos_dims / 2)));
        } else {
            encoding = cosf(row / powf(10000.0f, (2.0f * ((d - 1) / 2)) / (pos_dims / 2)));
        }
    } else if (d < pos_dims) {
        // Second part: column encoding
        int d_col = d - pos_dims / 2;
        if (d_col % 2 == 0) {
            encoding = sinf(col / powf(10000.0f, (2.0f * (d_col / 2)) / (pos_dims / 2)));
        } else {
            encoding = cosf(col / powf(10000.0f, (2.0f * ((d_col - 1) / 2)) / (pos_dims / 2)));
        }
    } else {
        // Third part: class label encoding
        int d_class = d - pos_dims;
        if (d_class % 2 == 0) {
            encoding = sinf(class_label / powf(10000.0f, (2.0f * (d_class / 2)) / class_dims));
        } else {
            encoding = cosf(class_label / powf(10000.0f, (2.0f * ((d_class - 1) / 2)) / class_dims));
        }
    }
    
    embedded[idx] += encoding;
}

__global__ void softmax_cross_entropy_row_kernel(float* loss_out, float* grad_logits, const float* logits, const unsigned char* targets, int rows, int vocab_size) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int base = row * vocab_size;

    extern __shared__ float shmem[];

    // Row-wise max for numerical stability
    float thread_max = -1e30f;
    for (int v = tid; v < vocab_size; v += stride) {
        float x = logits[base + v];
        thread_max = fmaxf(thread_max, x);
    }
    shmem[tid] = thread_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] = fmaxf(shmem[tid], shmem[tid + s]);
        __syncthreads();
    }
    float row_max = shmem[0];
    __syncthreads();

    // Row-wise sum of exp(logit - row_max), and stash exp in grad_logits
    float thread_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += stride) {
        float e = __expf(logits[base + v] - row_max);
        grad_logits[base + v] = e;
        thread_sum += e;
    }
    shmem[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    float row_sum = shmem[0];
    __syncthreads();

    // Normalize to probabilities and write gradient p - one_hot
    int t = targets[row];
    __shared__ float p_target;
    if (tid == 0) p_target = 0.0f;
    __syncthreads();

    for (int v = tid; v < vocab_size; v += stride) {
        float p = grad_logits[base + v] / row_sum;
        grad_logits[base + v] = p - (v == t ? 1.0f : 0.0f);
        if (v == t) p_target = p;
    }
    __syncthreads();

    // Accumulate the loss for this row: -log p(target)
    if (tid == 0) {
        float loss_row = -logf(fmaxf(p_target, 1e-30f));
        atomicAdd(loss_out, loss_row);
    }
}

// CUDA kernel for token embedding gradient accumulation
__global__ static void token_embedding_grad_kernel(float* token_embedding_grad, float* grad_embedded, unsigned char* tokens, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int token_idx = b * seq_len + t;
    int token = tokens[token_idx];
    int emb_idx = b * seq_len * d_model + t * d_model + d;
    
    atomicAdd(&token_embedding_grad[token * d_model + d], grad_embedded[emb_idx]);
}

// Forward pass
void forward_pass_sim(SIM* sim, unsigned char* d_input_tokens, unsigned char* d_class_labels) {
    // Step 1: Token embedding lookup
    dim3 grid_emb(sim->batch_size, sim->seq_len);
    dim3 block_emb(sim->d_model);
    token_embedding_lookup_kernel<<<grid_emb, block_emb>>>(
        sim->d_embedded_input, sim->d_token_embedding, d_input_tokens,
        sim->batch_size, sim->seq_len, sim->d_model
    );
    
    // Step 2: Add 2D positional encodings and class label conditioning
    positional_and_class_encoding_kernel<<<grid_emb, block_emb>>>(
        sim->d_embedded_input, d_class_labels, sim->batch_size, sim->seq_len, sim->d_model, sim->image_size
    );
    
    // Step 3: Forward pass through transformer
    forward_pass_transformer(sim->transformer, sim->d_embedded_input);
    
    // Step 4: Output projection through MLP
    forward_pass_mlp(sim->output_mlp, sim->transformer->mlp_layers[sim->num_layers-1]->d_output);
}

// Calculate loss
float calculate_loss_sim(SIM* sim, unsigned char* d_target_tokens) {
    // Reset loss accumulator
    CHECK_CUDA(cudaMemset(sim->d_loss_result, 0, sizeof(float)));
    
    // Compute softmax and cross-entropy loss
    softmax_cross_entropy_row_kernel<<<sim->batch_size * sim->seq_len, 256, 256 * sizeof(float)>>>(sim->d_loss_result, 
        sim->output_mlp->d_output, sim->output_mlp->d_output, d_target_tokens, sim->batch_size * sim->seq_len, sim->vocab_size);

    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, sim->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / (sim->batch_size * sim->seq_len);
}

// Zero gradients
void zero_gradients_sim(SIM* sim) {
    int token_emb_size = sim->vocab_size * sim->d_model;
    
    CHECK_CUDA(cudaMemset(sim->d_token_embedding_grad, 0, token_emb_size * sizeof(float)));
    
    zero_gradients_transformer(sim->transformer);
    zero_gradients_mlp(sim->output_mlp);
}

// Backward pass
void backward_pass_sim(SIM* sim, unsigned char* d_input_tokens, unsigned char* d_class_labels) {
    (void)d_class_labels; // Suppress unused parameter warning
    
    // Step 4 (backward): Backward pass through output MLP
    backward_pass_mlp(sim->output_mlp, 
                      sim->transformer->mlp_layers[sim->num_layers-1]->d_output, 
                      sim->transformer->mlp_layers[sim->num_layers-1]->d_output);
    
    // Step 3 (backward): Backward pass through transformer
    backward_pass_transformer(sim->transformer, sim->d_embedded_input, sim->d_embedded_input);
    
    // Step 2 (backward): Position and class encoding gradients pass through unchanged
    // (no learnable parameters, gradients flow through to token embeddings)
    
    // Step 1 (backward): Token embedding gradients
    dim3 grid_emb(sim->batch_size, sim->seq_len);
    dim3 block_emb(sim->d_model);
    
    token_embedding_grad_kernel<<<grid_emb, block_emb>>>(
        sim->d_token_embedding_grad, sim->d_embedded_input, d_input_tokens,
        sim->batch_size, sim->seq_len, sim->d_model
    );
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
    
    // Update token embeddings
    int token_emb_size = sim->vocab_size * sim->d_model;
    int token_blocks = (token_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_sim<<<token_blocks, block_size>>>(
        sim->d_token_embedding, sim->d_token_embedding_grad, sim->d_token_embedding_m, sim->d_token_embedding_v,
        sim->beta1, sim->beta2, sim->epsilon, learning_rate, sim->weight_decay,
        alpha_t, token_emb_size, sim->batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(sim->transformer, learning_rate);
    
    // Update output MLP weights
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
    fwrite(&sim->seq_len, sizeof(int), 1, file);
    fwrite(&sim->d_model, sizeof(int), 1, file);
    fwrite(&sim->batch_size, sizeof(int), 1, file);
    fwrite(&sim->hidden_dim, sizeof(int), 1, file);
    fwrite(&sim->num_layers, sizeof(int), 1, file);
    fwrite(&sim->vocab_size, sizeof(int), 1, file);
    fwrite(&sim->image_size, sizeof(int), 1, file);
    fwrite(&sim->num_classes, sizeof(int), 1, file);
    
    int token_emb_size = sim->vocab_size * sim->d_model;
    
    // Allocate host memory and copy embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding, sim->d_token_embedding, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding, sizeof(float), token_emb_size, file);
    
    // Save Adam state
    fwrite(&sim->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding_m, sim->d_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_token_embedding_v, sim->d_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(h_token_embedding_v, sizeof(float), token_emb_size, file);
    
    fclose(file);
    
    // Save transformer components
    char transformer_filename[256];
    char output_mlp_filename[256];
    char base_filename[256];
    
    // Remove .bin extension from filename to create base name
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    // Find and remove .bin extension if it exists
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    snprintf(output_mlp_filename, sizeof(output_mlp_filename), "%s_output_mlp.bin", base_filename);
    save_transformer(sim->transformer, transformer_filename);
    save_mlp(sim->output_mlp, output_mlp_filename);
    
    // Free temporary host memory
    free(h_token_embedding);
    free(h_token_embedding_m); free(h_token_embedding_v);

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
    int seq_len, d_model, stored_batch_size, hidden_dim, num_layers, vocab_size, image_size, num_classes;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&image_size, sizeof(int), 1, file);
    fread(&num_classes, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize SIM
    SIM* sim = init_sim(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    int token_emb_size = vocab_size * d_model;
    
    // Load embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    
    fread(h_token_embedding, sizeof(float), token_emb_size, file);
    
    CHECK_CUDA(cudaMemcpy(sim->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&sim->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    
    fread(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fread(h_token_embedding_v, sizeof(float), token_emb_size, file);
    
    CHECK_CUDA(cudaMemcpy(sim->d_token_embedding_m, h_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sim->d_token_embedding_v, h_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fclose(file);
    
    // Load transformer and output MLP components
    char transformer_filename[256];
    char output_mlp_filename[256];
    char base_filename[256];
    
    // Remove .bin extension from filename to create base name  
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    // Find and remove .bin extension if it exists
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    snprintf(output_mlp_filename, sizeof(output_mlp_filename), "%s_output_mlp.bin", base_filename);
    
    // Free the initialized components
    free_transformer(sim->transformer);
    free_mlp(sim->output_mlp);
    
    // Load the saved components
    sim->transformer = load_transformer(transformer_filename, batch_size, cublaslt_handle);
    sim->output_mlp = load_mlp(output_mlp_filename, batch_size * seq_len, cublaslt_handle);
    
    // Free temporary host memory
    free(h_token_embedding);
    free(h_token_embedding_m); free(h_token_embedding_v);
    
    printf("Model loaded from %s\n", filename);
    return sim;
}