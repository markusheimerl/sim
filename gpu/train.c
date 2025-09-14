#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include "sim.h"

SIM* model = NULL;

// SIGINT handler to save model and exit
void handle_sigint(int signum) {
    if (model) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_diffusion.bin", localtime(&now));
        //save_diffusion_model(model, model_filename);
    }
    exit(128 + signum);
}

int extract_cifar10_if_needed(const char* tar_path, const char* extract_dir) {
    struct stat st = {0};
    
    if (stat(extract_dir, &st) == 0) {
        printf("CIFAR-10 data already extracted in %s\n", extract_dir);
        return 1;
    }
    
    if (mkdir(extract_dir, 0755) != 0) {
        printf("Error: Could not create extraction directory: %s\n", extract_dir);
        return 0;
    }
    
    char extract_cmd[512];
    snprintf(extract_cmd, sizeof(extract_cmd), "tar -xzf %s -C %s --strip-components=1", tar_path, extract_dir);
    
    printf("Extracting CIFAR-10 data...\n");
    if (system(extract_cmd) != 0) {
        printf("Error: Failed to extract CIFAR-10 data\n");
        return 0;
    }
    
    printf("CIFAR-10 data extracted successfully\n");
    return 1;
}

void save_sample_image(SIM* model, float* d_sample, int sample_idx, const char* prefix) {
    // Convert patches back to image
    float* d_image;
    CHECK_CUDA(cudaMalloc(&d_image, 32 * 32 * 3 * sizeof(float)));
    
    patches_to_image(d_image, d_sample, 1);
    
    // Copy to host and convert to CIFAR format for saving
    float* h_image = (float*)malloc(32 * 32 * 3 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_image, d_image, 32 * 32 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    CIFAR10_Image sample_img;
    sample_img.label = 0;  // Unknown class for generated images
    
    for (int p = 0; p < 32 * 32 * 3; p++) {
        // Convert from [-1, 1] back to [0, 255] and rearrange to CIFAR format
        int pixel_idx = (p / 3) % (32 * 32);
        int channel = p % 3;
        int cifar_idx = channel * (32 * 32) + pixel_idx;
        
        float val = (h_image[p] + 1.0f) * 127.5f;  // Convert [-1, 1] to [0, 255]
        sample_img.pixels[cifar_idx] = (uint8_t)fmaxf(0.0f, fminf(255.0f, val));
    }
    
    char filename[256];
    snprintf(filename, sizeof(filename), "sample_images/%s_sample_%d.png", prefix, sample_idx);
    save_cifar10_image_png(&sample_img, filename);
    
    free(h_image);
    CHECK_CUDA(cudaFree(d_image));
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Paths
    const char* tar_path = "../cifar-10-binary.tar.gz";
    const char* extract_dir = "../cifar-10-data";
    
    // Extract CIFAR-10 data
    if (!extract_cifar10_if_needed(tar_path, extract_dir)) {
        return 1;
    }
    
    // Create output directory
    struct stat st = {0};
    if (stat("sample_images", &st) == -1) {
        mkdir("sample_images", 0755);
    }
    
    // Model parameters
    const int d_model = 256;
    const int hidden_dim = 512;
    const int num_layers = 6;
    const int batch_size = 8;
    
    // Load CIFAR-10 data
    char batch_path[256];
    snprintf(batch_path, sizeof(batch_path), "%s/data_batch_1.bin", extract_dir);
    
    CIFAR10_Dataset* dataset = load_cifar10_batch(batch_path);
    if (!dataset) {
        printf("Error: Could not load CIFAR-10 data\n");
        return 1;
    }
    
    print_cifar10_stats(dataset);
    
    // Initialize diffusion model
    model = init_diffusion_model(d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    printf("Diffusion model initialized:\n");
    printf("  Patch size: %dx%d\n", PATCH_SIZE, PATCH_SIZE);
    printf("  Number of patches: %d\n", NUM_PATCHES);
    printf("  Patch dimension: %d\n", PATCH_DIM);
    printf("  Model dimension: %d\n", d_model);
    printf("  Sequence length: %d (patches + time)\n", model->seq_len);
    
    // Prepare training data
    int num_batches = dataset->num_images / batch_size;
    printf("Training batches: %d\n", num_batches);
    
    // Allocate device memory for training
    float* d_images;
    float* d_clean_patches;
    float* d_noisy_patches;
    float* d_target_noise;
    int* d_timesteps;
    
    CHECK_CUDA(cudaMalloc(&d_images, batch_size * 32 * 32 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_clean_patches, batch_size * NUM_PATCHES * PATCH_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_noisy_patches, batch_size * NUM_PATCHES * PATCH_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_target_noise, batch_size * NUM_PATCHES * PATCH_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_timesteps, batch_size * sizeof(int)));
    
    // Training parameters
    const int num_epochs = 5;
    const float learning_rate = 0.0001f;
    
    printf("\nStarting training...\n");
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches && batch < 10; batch++) {  // Limit to 10 batches for demo
            // Prepare batch data on host
            float* h_images = (float*)malloc(batch_size * 32 * 32 * 3 * sizeof(float));
            int* h_timesteps = (int*)malloc(batch_size * sizeof(int));
            
            // Convert CIFAR-10 images to float format
            cifar10_to_float(h_images, &dataset->images[batch * batch_size], batch_size);
            
            // Random timesteps for each image in batch
            for (int i = 0; i < batch_size; i++) {
                h_timesteps[i] = rand() % MAX_TIMESTEPS;
            }
            
            // Copy to device
            CHECK_CUDA(cudaMemcpy(d_images, h_images, batch_size * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_timesteps, h_timesteps, batch_size * sizeof(int), cudaMemcpyHostToDevice));
            
            // Convert images to patches
            image_to_patches(d_clean_patches, d_images, batch_size);
            
            // Forward diffusion (add noise)
            forward_diffusion(model, d_noisy_patches, d_clean_patches, h_timesteps, batch_size);
            
            // Store target noise for loss calculation
            CHECK_CUDA(cudaMemcpy(d_target_noise, model->d_noise_pred, 
                                 batch_size * NUM_PATCHES * PATCH_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
            
            // Forward pass through model
            forward_pass_diffusion(model, d_noisy_patches, h_timesteps);
            
            // Calculate loss
            float loss = calculate_loss_diffusion(model, d_target_noise);
            epoch_loss += loss;
            
            // Backward pass and update
            zero_gradients_diffusion(model);
            backward_pass_diffusion(model, d_noisy_patches, h_timesteps);
            update_weights_diffusion(model, learning_rate);
            
            printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", 
                   epoch + 1, num_epochs, batch + 1, num_batches, loss);
            
            // Save sample every few batches
            if (batch % 5 == 0 && batch > 0) {
                printf("Saving training sample...\n");
                save_sample_image(model, d_clean_patches, batch, "original");
                save_sample_image(model, d_noisy_patches, batch, "noisy");
                save_sample_image(model, model->d_noise_pred, batch, "denoised");
            }
            
            free(h_images);
            free(h_timesteps);
        }
        
        epoch_loss /= (num_batches < 10 ? num_batches : 10);
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch + 1, num_epochs, epoch_loss);
    }
    
    printf("Training completed!\n");
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_images));
    CHECK_CUDA(cudaFree(d_clean_patches));
    CHECK_CUDA(cudaFree(d_noisy_patches));
    CHECK_CUDA(cudaFree(d_target_noise));
    CHECK_CUDA(cudaFree(d_timesteps));
    
    free_cifar10_dataset(dataset);
    free_diffusion_model(model);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}