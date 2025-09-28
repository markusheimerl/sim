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

// Convert float image [-1,1] to unsigned char tokens [0,255]
void float_to_tokens(float* image, unsigned char* tokens, int size) {
    for (int i = 0; i < size; i++) {
        // Clamp to [-1, 1] and convert to [0, 255]
        float val = fmaxf(-1.0f, fminf(1.0f, image[i]));
        tokens[i] = (unsigned char)((val + 1.0f) * 127.5f);
    }
}

// Convert unsigned char tokens [0,255] back to float image [-1,1]
void tokens_to_float(unsigned char* tokens, float* image, int size) {
    for (int i = 0; i < size; i++) {
        image[i] = (tokens[i] / 127.5f) - 1.0f;
    }
}

// Generate image function using autoregressive sampling with class conditioning
void generate_image(SIM* sim, float* generated_image, float temperature, unsigned char* d_input_tokens, unsigned char target_class) {
    const int image_pixels = 28 * 28;
    
    // Start with all zeros (black image)
    unsigned char* h_tokens = (unsigned char*)malloc(image_pixels * sizeof(unsigned char));
    memset(h_tokens, 0, image_pixels * sizeof(unsigned char));
    
    // Create class label array for batch (all same class)
    unsigned char* h_class_labels = (unsigned char*)malloc(sim->batch_size * sizeof(unsigned char));
    unsigned char* d_class_labels;
    CHECK_CUDA(cudaMalloc(&d_class_labels, sim->batch_size * sizeof(unsigned char)));
    
    for (int b = 0; b < sim->batch_size; b++) {
        h_class_labels[b] = target_class;
    }
    CHECK_CUDA(cudaMemcpy(d_class_labels, h_class_labels, sim->batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    printf("Generating class %d image pixel by pixel...\n", target_class);
    
    // Allocate logits buffer on host
    float* h_logits = (float*)malloc(sim->vocab_size * sizeof(float));
    
    // Generate pixels one at a time
    for (int pixel = 0; pixel < image_pixels - 1; pixel++) {
        // Copy current partial image to device (replicated across batch)
        for (int b = 0; b < sim->batch_size; b++) {
            CHECK_CUDA(cudaMemcpy(&d_input_tokens[b * image_pixels], h_tokens, image_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
        }
        
        // Forward pass with class conditioning
        forward_pass_sim(sim, d_input_tokens, d_class_labels);
        
        // Get logits for the current pixel position (use first batch element)
        CHECK_CUDA(cudaMemcpy(h_logits, &sim->output_mlp->d_output[pixel * sim->vocab_size], sim->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature and softmax
        float max_logit = -1e30f;
        for (int v = 0; v < sim->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < sim->vocab_size; v++) {
            float exp_val = expf(h_logits[v] - max_logit);
            h_logits[v] = exp_val;
            sum_exp += exp_val;
        }
        
        for (int v = 0; v < sim->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < sim->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Set the next pixel
        h_tokens[pixel + 1] = next_token;
        
        if (pixel % 100 == 0) {
            printf("Generated pixel %d/%d\n", pixel + 1, image_pixels);
        }
    }
    
    // Convert tokens back to float image
    tokens_to_float(h_tokens, generated_image, image_pixels);
    
    free(h_tokens);
    free(h_class_labels);
    free(h_logits);
    CHECK_CUDA(cudaFree(d_class_labels));
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

    // Parameters
    const int seq_len = 784;  // 28x28 pixels
    const int d_model = 384;
    const int hidden_dim = 1536;
    const int num_layers = 6;
    const int batch_size = 4;
    
    // Load MNIST data
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
    
    // Convert float images to token sequences
    unsigned char* input_tokens = (unsigned char*)malloc(num_images * seq_len * sizeof(unsigned char));
    unsigned char* target_tokens = (unsigned char*)malloc(num_images * seq_len * sizeof(unsigned char));
    
    for (int img = 0; img < num_images; img++) {
        float_to_tokens(&mnist_images[img * seq_len], &input_tokens[img * seq_len], seq_len);
        // Target is shifted by one pixel
        memcpy(&target_tokens[img * seq_len], &input_tokens[img * seq_len] + 1, (seq_len - 1) * sizeof(unsigned char));
        // Pad last target with 0
        target_tokens[img * seq_len + seq_len - 1] = 0;
    }
    
    // Save some sample images with their labels to verify loading
    printf("Saving sample images with labels to verify data loading...\n");
    for (int i = 0; i < 10; i++) {
        char sample_filename[256];
        snprintf(sample_filename, sizeof(sample_filename), "generated_images/sample_label_%d_idx_%d.png", mnist_labels[i], i);
        save_mnist_image_png(&mnist_images[i * seq_len], sample_filename);
        printf("Saved sample %d: label=%d, file=%s\n", i, mnist_labels[i], sample_filename);
    }
    
    // Initialize or load SIM
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        sim = load_sim(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("Initializing new model...\n");
        sim = init_sim(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Total parameters: ~%.1fM\n", (float)(sim->vocab_size * d_model + d_model * sim->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Training parameters
    const int num_epochs = 20;
    const float learning_rate = 0.0001f;
    const int num_batches = num_images / batch_size;

    // Allocate device memory for batch data
    unsigned char *d_input_tokens, *d_target_tokens, *d_class_labels;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_class_labels, batch_size * sizeof(unsigned char)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_class_labels, &mnist_labels[batch * batch_size], batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            // Forward pass with class conditioning
            forward_pass_sim(sim, d_input_tokens, d_class_labels);
            
            // Calculate loss
            float loss = calculate_loss_sim(sim, d_target_tokens);
            if(loss >= 8.0) raise(SIGINT);
            
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_sim(sim);
            backward_pass_sim(sim, d_input_tokens, d_class_labels);
            
            // Update weights
            update_weights_sim(sim, learning_rate);
            
            // Print progress
            if (batch % 10 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            }
            
            // Generate sample images periodically with specific class conditioning
            if (batch > 0 && batch % 500 == 0) {
                printf("\n--- Generating sample image (epoch %d, batch %d) ---\n", epoch, batch);
                
                // Get a random class from current batch
                int batch_img_idx = batch * batch_size + (rand() % batch_size);
                unsigned char target_class = mnist_labels[batch_img_idx];
                
                float* generated_image = (float*)malloc(seq_len * sizeof(float));
                generate_image(sim, generated_image, 0.8f, d_input_tokens, target_class);
                
                // Save generated image with target class
                char gen_filename[256];
                snprintf(gen_filename, sizeof(gen_filename), "generated_images/generated_epoch_%d_batch_%d_class_%d.png", epoch, batch, target_class);
                save_mnist_image_png(generated_image, gen_filename);
                
                printf("Saved generated image (target class %d): %s\n", target_class, gen_filename);
                printf("--- End generation ---\n\n");
                
                free(generated_image);
            }

            // Checkpoint model periodically
            if (batch > 0 && batch % 2000 == 0) {
                char checkpoint_fname[64];
                snprintf(checkpoint_fname, sizeof(checkpoint_fname), "checkpoint_sim.bin");
                save_sim(sim, checkpoint_fname);
            }
        }
        
        epoch_loss /= num_batches;

        // Print epoch summary
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch, num_epochs, epoch_loss);
        
        // Generate sample at end of each epoch for each digit class
        if (epoch < num_epochs) {
            printf("Generating end-of-epoch samples for each digit...\n");
            for (int digit = 0; digit < 10; digit++) {
                float* generated_image = (float*)malloc(seq_len * sizeof(float));
                generate_image(sim, generated_image, 0.7f, d_input_tokens, (unsigned char)digit);
                
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

    // Save model with timestamped filename
    save_sim(sim, model_fname);
    
    // Generate final samples for each digit with different temperatures
    printf("\nGenerating final samples for each digit...\n");
    for (int digit = 0; digit < 10; digit++) {
        for (int temp_idx = 0; temp_idx < 3; temp_idx++) {
            float temperature = 0.5f + temp_idx * 0.2f;
            float* generated_image = (float*)malloc(seq_len * sizeof(float));
            generate_image(sim, generated_image, temperature, d_input_tokens, (unsigned char)digit);
            
            char gen_filename[256];
            snprintf(gen_filename, sizeof(gen_filename), "generated_images/final_class_%d_temp_%.1f_sample_%d.png", digit, temperature, temp_idx);
            save_mnist_image_png(generated_image, gen_filename);
            printf("Saved final class %d sample (temp %.1f): %s\n", digit, temperature, gen_filename);
            
            free(generated_image);
        }
    }
    
    // Cleanup
    free(mnist_images);
    free(mnist_labels);
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    CHECK_CUDA(cudaFree(d_class_labels));
    free_sim(sim);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}