#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include "sim.h"

SIM* sim = NULL;

void handle_sigint(int sig) {
    (void)sig;
    if (sim) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_sim.bin", localtime(&now));
        //save_sim(sim, model_filename);
    }
    exit(0);
}

void cifar10_to_float(float* output, const CIFAR10_Image* images, int num_images) {
    for (int i = 0; i < num_images; i++) {
        for (int p = 0; p < 32 * 32; p++) {
            int out_idx = i * (32 * 32 * 3) + p * 3;
            // Normalize to [-1, 1]
            output[out_idx + 0] = (images[i].pixels[p] / 127.5f) - 1.0f;
            output[out_idx + 1] = (images[i].pixels[p + 32*32] / 127.5f) - 1.0f;
            output[out_idx + 2] = (images[i].pixels[p + 32*32*2] / 127.5f) - 1.0f;
        }
    }
}

void patches_to_image(float* patches, CIFAR10_Image* image, int sample_idx) {
    for (int patch_idx = 0; patch_idx < NUM_PATCHES; patch_idx++) {
        int patch_row = patch_idx / PATCHES_PER_ROW;
        int patch_col = patch_idx % PATCHES_PER_ROW;
        
        for (int pixel_in_patch = 0; pixel_in_patch < PATCH_DIM; pixel_in_patch++) {
            int pixel_idx = pixel_in_patch / 3;
            int channel = pixel_in_patch % 3;
            
            int pixel_row = pixel_idx / PATCH_SIZE;
            int pixel_col = pixel_idx % PATCH_SIZE;
            
            int global_row = patch_row * PATCH_SIZE + pixel_row;
            int global_col = patch_col * PATCH_SIZE + pixel_col;
            
            // Denormalize from [-1, 1] to [0, 255]
            float val = (patches[sample_idx * NUM_PATCHES * PATCH_DIM + patch_idx * PATCH_DIM + pixel_in_patch] + 1.0f) * 127.5f;
            val = fmaxf(0.0f, fminf(255.0f, val));
            
            image->pixels[channel * 32 * 32 + global_row * 32 + global_col] = (uint8_t)val;
        }
    }
    image->label = 0;
}

void save_sample_images(SIM* sim, int epoch, int batch) {
    // Copy patches back to host
    int patch_buffer_size = sim->batch_size * NUM_PATCHES * PATCH_DIM;
    float* h_clean = (float*)malloc(patch_buffer_size * sizeof(float));
    float* h_noisy = (float*)malloc(patch_buffer_size * sizeof(float));
    float* h_denoised = (float*)malloc(patch_buffer_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_clean, sim->d_clean_patches, patch_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_noisy, sim->d_noisy_patches, patch_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_denoised, sim->output_mlp->d_layer_output, patch_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Save first sample from batch
    CIFAR10_Image clean_img, noisy_img, denoised_img;
    
    patches_to_image(h_clean, &clean_img, 0);
    patches_to_image(h_noisy, &noisy_img, 0);
    patches_to_image(h_denoised, &denoised_img, 0);
    
    char filename[256];
    snprintf(filename, sizeof(filename), "sample_images/epoch_%d_batch_%d_clean.png", epoch, batch);
    save_cifar10_image_png(&clean_img, filename);
    
    snprintf(filename, sizeof(filename), "sample_images/epoch_%d_batch_%d_noisy.png", epoch, batch);
    save_cifar10_image_png(&noisy_img, filename);
    
    snprintf(filename, sizeof(filename), "sample_images/epoch_%d_batch_%d_pred.png", epoch, batch);
    save_cifar10_image_png(&denoised_img, filename);
    
    free(h_clean);
    free(h_noisy);
    free(h_denoised);
}

void generate_and_save_samples(SIM* sim, int epoch) {
    const int num_gen_samples = 4;
    const int num_inference_steps = 50;
    
    printf("\n=== Generating %d sample images at epoch %d ===\n", num_gen_samples, epoch);
    
    // Allocate memory for generated images
    float* d_generated_images;
    CHECK_CUDA(cudaMalloc(&d_generated_images, num_gen_samples * 32 * 32 * 3 * sizeof(float)));
    
    // Generate images
    generate_images_sim(sim, d_generated_images, num_gen_samples, num_inference_steps);
    
    // Copy to host
    float* h_generated_images = (float*)malloc(num_gen_samples * 32 * 32 * 3 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_generated_images, d_generated_images, 
                         num_gen_samples * 32 * 32 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Debug: Print some pixel values
    printf("Sample pixel values: [%.3f, %.3f, %.3f, %.3f, %.3f]\n", 
           h_generated_images[0], h_generated_images[1], h_generated_images[2], 
           h_generated_images[3], h_generated_images[4]);
    
    // Convert to CIFAR format and save
    for (int i = 0; i < num_gen_samples; i++) {
        CIFAR10_Image generated_img;
        
        // Convert from interleaved RGB to CIFAR format
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int pixel_idx = y * 32 + x;
                int rgb_idx = i * (32 * 32 * 3) + pixel_idx * 3;
                
                // Clamp to [-1, 1] range first
                float r = fmaxf(-1.0f, fminf(1.0f, h_generated_images[rgb_idx + 0]));
                float g = fmaxf(-1.0f, fminf(1.0f, h_generated_images[rgb_idx + 1]));
                float b = fmaxf(-1.0f, fminf(1.0f, h_generated_images[rgb_idx + 2]));
                
                // Convert to [0, 255]
                generated_img.pixels[pixel_idx] = (uint8_t)((r + 1.0f) * 127.5f);
                generated_img.pixels[pixel_idx + 32*32] = (uint8_t)((g + 1.0f) * 127.5f);
                generated_img.pixels[pixel_idx + 32*32*2] = (uint8_t)((b + 1.0f) * 127.5f);
            }
        }
        
        generated_img.label = 0;
        
        char filename[256];
        snprintf(filename, sizeof(filename), "sample_images/generated_epoch_%d_sample_%d.png", epoch, i);
        save_cifar10_image_png(&generated_img, filename);
        
        printf("Saved: %s\n", filename);
    }
    
    printf("Image generation completed!\n\n");
    
    // Cleanup
    free(h_generated_images);
    CHECK_CUDA(cudaFree(d_generated_images));
}

int main(void) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    const int d_model = 512;
    const int hidden_dim = 2048;
    const int num_layers = 8;
    const int batch_size = 32;
    const char* cifar_path = "../cifar-10-batches-bin/data_batch_1.bin";
    
    // Create output directory
    struct stat st;
    if (stat("sample_images", &st) == -1) {
        mkdir("sample_images", 0755);
    }
    
    // Load CIFAR-10 data
    CIFAR10_Dataset* dataset = load_cifar10_batch(cifar_path);
    if (!dataset) {
        printf("Error: Could not load CIFAR-10 data\n");
        return 1;
    }
    print_cifar10_stats(dataset);
    
    // Initialize SIM
    sim = init_sim(d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    printf("SIM initialized:\n");
    printf("  Patch size: %dx%d, Patches: %d, Patch dim: %d\n", PATCH_SIZE, PATCH_SIZE, NUM_PATCHES, PATCH_DIM);
    printf("  Model dim: %d, Layers: %d\n", d_model, num_layers);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    const int num_batches = dataset->num_images / batch_size;
    
    // Allocate memory
    float* d_images;
    CHECK_CUDA(cudaMalloc(&d_images, batch_size * 32 * 32 * 3 * sizeof(float)));
    float* h_images = (float*)malloc(batch_size * 32 * 32 * 3 * sizeof(float));
    
    printf("\nStarting training...\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Prepare batch
            cifar10_to_float(h_images, &dataset->images[batch * batch_size], batch_size);
            CHECK_CUDA(cudaMemcpy(d_images, h_images, batch_size * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));
            
            // Random timestep (more early timesteps for stable training)
            int timestep = (int)(powf((float)rand() / (float)RAND_MAX, 2.0f) * MAX_TIMESTEPS);
            timestep = fminf(timestep, MAX_TIMESTEPS - 1);
            
            // Forward pass
            forward_pass_sim(sim, d_images, timestep);
            
            // Calculate loss
            float loss = calculate_loss_sim(sim);
            epoch_loss += loss;
            
            // Backward pass
            zero_gradients_sim(sim);
            backward_pass_sim(sim, timestep);
            update_weights_sim(sim, learning_rate);
            
            // Print progress
            if (batch % 2 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Timestep [%d], Loss: %.6f\n",
                       epoch + 1, num_epochs, batch + 1, num_batches, timestep, loss);
            }
            
            // Save training samples
            if (batch % 400 == 0 && batch > 0) {
                save_sample_images(sim, epoch, batch);
            }
        }
        
        epoch_loss /= num_batches;
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch + 1, num_epochs, epoch_loss);
        
        // Generate sample images every 2 epochs
        if ((epoch + 1) % 2 == 0) {
            generate_and_save_samples(sim, epoch + 1);
        }
        
        // Save checkpoint
        if ((epoch + 1) % 3 == 0) {
            save_sim(sim, "checkpoint_sim.bin");
            printf("Checkpoint saved\n");
        }
    }
    
    printf("Training completed!\n");
    
    // Save final model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_sim.bin", localtime(&now));
    save_sim(sim, model_filename);
    
    // Final generation
    generate_and_save_samples(sim, num_epochs);
    
    // Cleanup
    free(h_images);
    CHECK_CUDA(cudaFree(d_images));
    free_cifar10_dataset(dataset);
    free_sim(sim);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}