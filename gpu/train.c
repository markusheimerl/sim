#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <curand.h>
#include "../data.h"
#include "sim.h"

SIM* sim = NULL;

// SIGINT handler to save model and exit
void handle_sigint(int signum) {
    if (sim) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_sim.bin", localtime(&now));
        save_sim(sim, model_filename);
    }
    exit(128 + signum);
}

// Kernel: x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
__global__ static void add_noise_kernel(float* x_t, const float* x0, const float* noise,
                                        const float* sqrt_alpha_bar, const float* sqrt_one_minus_alpha_bar,
                                        int t_idx, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float a = sqrt_alpha_bar[t_idx];
    float b = sqrt_one_minus_alpha_bar[t_idx];
    x_t[i] = a * x0[i] + b * noise[i];
}

// Kernel: DDPM step
// x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_hat ) + sqrt(beta_t) * z
__global__ static void ddpm_step_kernel(float* x_prev, const float* x_t, const float* eps_hat, const float* z,
                                        const float* alphas, const float* sqrt_one_minus_alpha_bar, const float* sqrt_recip_alphas, const float* sqrt_betas,
                                        int t_idx, int total, int add_noise) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    
    float alpha_t = alphas[t_idx];
    float one_minus_alpha_t = 1.0f - alpha_t;
    float coef = one_minus_alpha_t / sqrt_one_minus_alpha_bar[t_idx];
    float term = sqrt_recip_alphas[t_idx] * (x_t[i] - coef * eps_hat[i]);
    float noise = add_noise ? sqrt_betas[t_idx] * z[i] : 0.0f;
    x_prev[i] = term + noise;
}

// Generate a single image via DDPM sampling, conditioned on class label
static void generate_image(SIM* sim, float* generated_image, float temperature,
                           unsigned char* d_class_labels, curandGenerator_t gen) {
    (void)temperature; // kept for API compatibility; not used in DDPM sampling
    
    const int image_pixels = sim->seq_len;
    const int total = sim->seq_len * sim->batch_size;
    
    // Working buffers
    float* d_x;       // current x_t
    float* d_x_prev;  // x_{t-1}
    float* d_z;       // random normal
    CHECK_CUDA(cudaMalloc(&d_x, sim->batch_size * sim->seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x_prev, sim->batch_size * sim->seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z, sim->batch_size * sim->seq_len * sizeof(float)));
    
    // Start from pure Gaussian noise
    curandGenerateNormal(gen, d_x, sim->batch_size * sim->seq_len, 0.0f, 1.0f);
    
    int block = 256;
    int grid = (sim->batch_size * sim->seq_len + block - 1) / block;
    
    // Iterate from T to 1
    for (int t = sim->T; t >= 1; t--) {
        // Predict epsilon at time t
        forward_pass_sim(sim, d_x, d_class_labels, t);
        
        // Copy predicted epsilon: output head has shape [B*seq_len * 1], contiguous
        // We'll use the same buffer pointer directly
        float* d_eps_hat = sim->output_mlp->d_layer_output;
        
        // Sample z for next step (except for t=1)
        if (t > 1) {
            curandGenerateNormal(gen, d_z, sim->batch_size * sim->seq_len, 0.0f, 1.0f);
        }
        
        // DDPM step: x_{t-1}
        ddpm_step_kernel<<<grid, block>>>(
            d_x_prev, d_x, d_eps_hat, d_z,
            sim->d_alphas, sim->d_sqrt_one_minus_alphas_cumprod, sim->d_sqrt_recip_alphas, sim->d_sqrt_betas,
            t - 1, sim->batch_size * sim->seq_len, (t > 1 ? 1 : 0)
        );
        
        // Swap buffers: d_x <= d_x_prev
        float* tmp = d_x;
        d_x = d_x_prev;
        d_x_prev = tmp;
    }
    
    // After loop, d_x holds x_0 (for batch); copy first image to host
    CHECK_CUDA(cudaMemcpy(generated_image, d_x, image_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_x_prev));
    CHECK_CUDA(cudaFree(d_z));
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Create output directory
    struct stat st;
    if (stat("generated_images", &st) == -1) {
        mkdir("generated_images", 0755);
    }

    // Parameters (can be tuned)
    const int seq_len = 784;   // 28x28 pixels
    const int d_model = 384;
    const int hidden_dim = 1536;
    const int num_layers = 6;
    const int batch_size = 4;
    const int T = 1000;        // diffusion steps
    
    // Load MNIST data (float images in [-1, 1])
    float* mnist_images = NULL;
    int num_images = 0;
    load_mnist_data(&mnist_images, &num_images, "../train-images-idx3-ubyte");
    if (!mnist_images) {
        printf("Error: Could not load MNIST data\n");
        return 1;
    }
    
    // Load MNIST labels
    unsigned char* mnist_labels = NULL;
    int num_labels = 0;
    load_mnist_labels(&mnist_labels, &num_labels, "../train-labels-idx1-ubyte");
    if (!mnist_labels) {
        printf("Error: Could not load MNIST labels\n");
        free(mnist_images);
        return 1;
    }
    
    // Verify data and labels match
    if (num_images != num_labels) {
        printf("Error: Number of images (%d) doesn't match number of labels (%d)\n", num_images, num_labels);
        free(mnist_images);
        free(mnist_labels);
        return 1;
    }
    
    printf("Data loaded successfully: %d images with matching labels\n", num_images);
    
    // Save some sample images with their labels to verify loading
    printf("Saving sample images with labels to verify data loading...\n");
    for (int i = 0; i < 10; i++) {
        char sample_filename[256];
        snprintf(sample_filename, sizeof(sample_filename), "generated_images/sample_label_%d_idx_%d.png", mnist_labels[i], i);
        save_mnist_image_png(&mnist_images[i * seq_len], sample_filename);
        printf("Saved sample %d: label=%d, file=%s\n", i, mnist_labels[i], sample_filename);
    }
    
    // Initialize or load SIM (Diffusion model)
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        sim = load_sim(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("Initializing new diffusion model...\n");
        sim = init_sim(seq_len, d_model, hidden_dim, num_layers, batch_size, T, cublaslt_handle);
    }
    
    // Rough parameter count estimate
    float approx_params =
        (float)(d_model * 2) +                                    // input projection w & b
        (float)(num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) + // transformer
        (float)(d_model * hidden_dim + hidden_dim * 1);           // output mlp
    printf("Approx total parameters: ~%.1fM\n", approx_params / 1e6f);
    
    // Training parameters
    const int num_epochs = 20;
    const float learning_rate = 0.0001f;
    const int num_batches = num_images / batch_size;

    // cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)time(NULL));
    
    // Allocate device memory for batch data
    float *d_x0, *d_noise, *d_x_t;
    unsigned char *d_class_labels;
    CHECK_CUDA(cudaMalloc(&d_x0, batch_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_noise, batch_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x_t, batch_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_class_labels, batch_size * sizeof(unsigned char)));
    
    int block = 256;
    int grid = (batch_size * seq_len + block - 1) / block;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int img_offset = batch * batch_size * seq_len;
            int lbl_offset = batch * batch_size;
            
            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_x0, &mnist_images[img_offset], batch_size * seq_len * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_class_labels, &mnist_labels[lbl_offset], batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            // Sample timestep t uniform in [1, T]
            int t = (rand() % sim->T) + 1;
            
            // Sample noise
            curandGenerateNormal(gen, d_noise, batch_size * seq_len, 0.0f, 1.0f);
            
            // Compute x_t
            add_noise_kernel<<<grid, block>>>(
                d_x_t, d_x0, d_noise,
                sim->d_sqrt_alphas_cumprod, sim->d_sqrt_one_minus_alphas_cumprod,
                t - 1, batch_size * seq_len
            );
            
            // Forward pass (predict epsilon)
            forward_pass_sim(sim, d_x_t, d_class_labels, t);
            
            // Loss: MSE between predicted epsilon and true noise
            float loss = calculate_loss_sim(sim, d_noise);
            epoch_loss += loss;
            
            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;
            
            // Backward & update
            zero_gradients_sim(sim);
            backward_pass_sim(sim, d_x_t);
            update_weights_sim(sim, learning_rate);
            
            if (batch % 50 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], t=%d, Loss: %.6f\n", epoch, num_epochs, batch, num_batches, t, loss);
            }
            
            // Generate occasional sample during training
            if (batch > 0 && batch % 1000 == 0) {
                printf("\n--- Generating sample image (epoch %d, batch %d) ---\n", epoch, batch);
                // Pick a random class from current batch
                int batch_img_idx = batch * batch_size + (rand() % batch_size);
                unsigned char target_class = mnist_labels[batch_img_idx];
                
                unsigned char* h_class = (unsigned char*)malloc(batch_size * sizeof(unsigned char));
                for (int b = 0; b < batch_size; b++) h_class[b] = target_class;
                CHECK_CUDA(cudaMemcpy(d_class_labels, h_class, batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
                free(h_class);
                
                float* generated_image = (float*)malloc(seq_len * sizeof(float));
                generate_image(sim, generated_image, 1.0f, d_class_labels, gen);
                
                char gen_filename[256];
                snprintf(gen_filename, sizeof(gen_filename), "generated_images/generated_epoch_%d_batch_%d_class_%d.png", epoch, batch, target_class);
                save_mnist_image_png(generated_image, gen_filename);
                printf("Saved generated image (target class %d): %s\n", target_class, gen_filename);
                printf("--- End generation ---\n\n");
                
                free(generated_image);
            }
            
            // Checkpoint
            if (batch > 0 && batch % 2000 == 0) {
                char checkpoint_fname[64];
                snprintf(checkpoint_fname, sizeof(checkpoint_fname), "checkpoint_sim.bin");
                save_sim(sim, checkpoint_fname);
            }
        }
        
        epoch_loss /= num_batches;
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch, num_epochs, epoch_loss);
        
        // Generate end-of-epoch samples for each digit
        if (epoch < num_epochs) {
            printf("Generating end-of-epoch samples for each digit...\n");
            for (int digit = 0; digit < 10; digit++) {
                unsigned char* h_class = (unsigned char*)malloc(batch_size * sizeof(unsigned char));
                for (int b = 0; b < batch_size; b++) h_class[b] = (unsigned char)digit;
                CHECK_CUDA(cudaMemcpy(d_class_labels, h_class, batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
                free(h_class);
                
                float* generated_image = (float*)malloc(seq_len * sizeof(float));
                generate_image(sim, generated_image, 1.0f, d_class_labels, gen);
                
                char gen_filename[256];
                snprintf(gen_filename, sizeof(gen_filename), "generated_images/epoch_%d_class_%d_sample.png", epoch, digit);
                save_mnist_image_png(generated_image, gen_filename);
                printf("Saved epoch %d class %d sample: %s\n", epoch, digit, gen_filename);
                
                free(generated_image);
            }
        }
    }

    // Get timestamp for filenames
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_sim.bin", localtime(&now));

    // Save final model
    save_sim(sim, model_fname);
    
    // Generate final samples for each digit (using DDPM; temperature unused)
    printf("\nGenerating final samples for each digit...\n");
    for (int digit = 0; digit < 10; digit++) {
        unsigned char* h_class = (unsigned char*)malloc(batch_size * sizeof(unsigned char));
        for (int b = 0; b < batch_size; b++) h_class[b] = (unsigned char)digit;
        CHECK_CUDA(cudaMemcpy(d_class_labels, h_class, batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        free(h_class);
        
        for (int sample_idx = 0; sample_idx < 3; sample_idx++) {
            float* generated_image = (float*)malloc(seq_len * sizeof(float));
            generate_image(sim, generated_image, 1.0f, d_class_labels, gen);
            
            char gen_filename[256];
            snprintf(gen_filename, sizeof(gen_filename), "generated_images/final_class_%d_sample_%d.png", digit, sample_idx);
            save_mnist_image_png(generated_image, gen_filename);
            printf("Saved final class %d sample: %s\n", digit, gen_filename);
            
            free(generated_image);
        }
    }
    
    // Cleanup
    free(mnist_images);
    free(mnist_labels);
    CHECK_CUDA(cudaFree(d_x0));
    CHECK_CUDA(cudaFree(d_noise));
    CHECK_CUDA(cudaFree(d_x_t));
    CHECK_CUDA(cudaFree(d_class_labels));
    curandDestroyGenerator(gen);
    free_sim(sim);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}