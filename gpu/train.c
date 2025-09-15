#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include "sim.h"

SIM* sim = NULL;

void handle_sigint(int sig) {
    (void)sig;
    if (sim) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_sim.bin", localtime(&now));
        save_sim(sim, model_filename);
    }
    exit(0);
}

void cifar10_to_float(float* output, const CIFAR10_Image* images, int num_images) {
    for (int i = 0; i < num_images; i++) {
        for (int p = 0; p < 32 * 32; p++) {
            // Convert CIFAR format (R[1024], G[1024], B[1024]) to interleaved RGB normalized to [-1,1]
            int out_idx = i * (32 * 32 * 3) + p * 3;
            output[out_idx + 0] = (images[i].pixels[p] / 127.5f) - 1.0f;              // R
            output[out_idx + 1] = (images[i].pixels[p + 32*32] / 127.5f) - 1.0f;      // G
            output[out_idx + 2] = (images[i].pixels[p + 32*32*2] / 127.5f) - 1.0f;    // B
        }
    }
}

void save_sample_image(float* d_output_patches, int sample_idx, const char* prefix) {
    // Copy patches back to host
    float* h_patches = (float*)malloc(NUM_PATCHES * PATCH_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_patches, d_output_patches, NUM_PATCHES * PATCH_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert patches to image
    CIFAR10_Image sample_img;
    sample_img.label = 0;  // Unknown class for generated images
    
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
            
            float val = (h_patches[patch_idx * PATCH_DIM + pixel_in_patch] + 1.0f) * 127.5f;
            val = fmaxf(0.0f, fminf(255.0f, val));
            
            // CIFAR format: R[1024], G[1024], B[1024]
            sample_img.pixels[channel * 32 * 32 + global_row * 32 + global_col] = (uint8_t)val;
        }
    }
    
    char filename[256];
    snprintf(filename, sizeof(filename), "sample_images/%s_sample_%d.png", prefix, sample_idx);
    save_cifar10_image_png(&sample_img, filename);
    
    free(h_patches);
}

int main(void) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    const int d_model = 256;
    const int hidden_dim = 512;
    const int num_layers = 6;
    const int batch_size = 8;
    const char* cifar_path = "../cifar-10-data/data_batch_1.bin";
    
    // Create output directory
    struct stat st;
    memset(&st, 0, sizeof(st));
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
    
    printf("Small Image Model (SIM) initialized:\n");
    printf("  Patch size: %dx%d\n", PATCH_SIZE, PATCH_SIZE);
    printf("  Number of patches: %d\n", NUM_PATCHES);
    printf("  Patch dimension: %d\n", PATCH_DIM);
    printf("  Model dimension: %d\n", d_model);
    printf("  Transformer sequence length: %d (patches only)\n", NUM_PATCHES);
    
    // Training parameters
    const int num_epochs = 20;
    const float learning_rate = 0.0001f;
    const int num_batches = dataset->num_images / batch_size;
    
    // Allocate device memory for training
    float* d_images;
    CHECK_CUDA(cudaMalloc(&d_images, batch_size * 32 * 32 * 3 * sizeof(float)));
    
    // Allocate host memory for batch
    float* h_images = (float*)malloc(batch_size * 32 * 32 * 3 * sizeof(float));
    
    printf("\nStarting training...\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int batches_processed = 0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Prepare batch data
            cifar10_to_float(h_images, &dataset->images[batch * batch_size], batch_size);
            
            // Copy to device
            CHECK_CUDA(cudaMemcpy(d_images, h_images, batch_size * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));
            
            // Random timestep
            int timestep = rand() % MAX_TIMESTEPS;
            
            // Forward pass
            forward_pass_sim(sim, d_images, timestep);
            
            // Calculate loss
            float loss = calculate_loss_sim(sim);
            epoch_loss += loss;
            batches_processed++;
            
            // Backward pass and update
            zero_gradients_sim(sim);
            backward_pass_sim(sim, timestep);
            update_weights_sim(sim, learning_rate);
            
            // Print progress every 100 batches
            if (batch % 100 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Timestep [%d], Loss: %.6f\n", 
                       epoch + 1, num_epochs, batch + 1, num_batches, timestep, loss);
            }
            
            // Save sample every 200 batches
            if (batch % 200 == 0 && batch > 0) {
                printf("Saving training samples...\n");
                save_sample_image(sim->output_mlp->d_layer_output, batch, "denoised");
                
                // Also save the original and noisy versions for comparison
                save_sample_image(sim->d_patches, batch, "original");
                save_sample_image(sim->d_noisy_patches, batch, "noisy");
            }
        }
        
        epoch_loss /= batches_processed;
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch + 1, num_epochs, epoch_loss);
        
        // Save checkpoint every few epochs
        if ((epoch + 1) % 5 == 0) {
            char checkpoint_filename[64];
            snprintf(checkpoint_filename, sizeof(checkpoint_filename), "checkpoint_sim.bin");
            save_sim(sim, checkpoint_filename);
            printf("Checkpoint saved\n");
        }
    }
    
    printf("Training completed!\n");
    
    // Save final model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_sim.bin", localtime(&now));
    save_sim(sim, model_filename);
    
    // Cleanup
    free(h_images);
    CHECK_CUDA(cudaFree(d_images));
    free_cifar10_dataset(dataset);
    free_sim(sim);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}