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

// Embed class information into first pixel and free labels
void embed_class_in_first_pixel(float* mnist_images, unsigned char* mnist_labels, int num_images) {
    for (int img = 0; img < num_images; img++) {
        // Map class 0-9 to dark values: 0->-1.0, 1->-0.9, ..., 9->-0.1
        float class_value = -1.0f + (mnist_labels[img] * 0.1f);
        mnist_images[img * 784] = class_value; // Overwrite first pixel
    }
    printf("Embedded class information into first pixel of all images\n");
}

// Extract class from first pixel for generation
unsigned char extract_class_from_first_pixel(float first_pixel_value) {
    // Inverse of the embedding: class = (value + 1.0) / 0.1
    float class_float = (first_pixel_value + 1.0f) / 0.1f;
    return (unsigned char)round(fmaxf(0.0f, fminf(9.0f, class_float)));
}

// Generate image function using autoregressive sampling
void generate_image(SIM* sim, float* generated_image, float temperature, unsigned char* d_input_tokens, unsigned char target_class) {
    const int image_pixels = 28 * 28;
    
    // Start with all zeros (black image)
    unsigned char* h_tokens = (unsigned char*)malloc(image_pixels * sizeof(unsigned char));
    memset(h_tokens, 0, image_pixels * sizeof(unsigned char));
    
    // Set first pixel to target class
    float class_value = -1.0f + (target_class * 0.1f);
    h_tokens[0] = (unsigned char)((class_value + 1.0f) * 127.5f);
    
    printf("Generating class %d image pixel by pixel...\n", target_class);
    
    // Allocate logits buffer on host
    float* h_logits = (float*)malloc(sim->vocab_size * sizeof(float));
    
    // Generate pixels one at a time (starting from pixel 1, since pixel 0 is the class)
    for (int pixel = 0; pixel < image_pixels - 1; pixel++) {
        // Copy current partial image to device (replicated across batch)
        for (int b = 0; b < sim->batch_size; b++) {
            CHECK_CUDA(cudaMemcpy(&d_input_tokens[b * image_pixels], h_tokens, image_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
        }
        
        // Forward pass
        forward_pass_sim(sim, d_input_tokens);
        
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
    free(h_logits);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len = 784;  // 28x28 pixels
    const int d_model = 512;
    const int hidden_dim = 2048;
    const int num_layers = 12;
    const int batch_size = 64;
    
    // Load MNIST data and labels
    float* mnist_images = NULL;
    unsigned char* mnist_labels = NULL;
    int num_images, num_labels = 0;
    load_mnist_data(&mnist_images, &num_images, "../train-images-idx3-ubyte");
    load_mnist_labels(&mnist_labels, &num_labels, "../train-labels-idx1-ubyte");

    // Embed class information into first pixel
    embed_class_in_first_pixel(mnist_images, mnist_labels, num_images);
    
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
    
    // Save some sample images to verify preprocessing
    printf("Saving sample images to verify class embedding...\n");
    for (int i = 0; i < 10; i++) {
        unsigned char first_pixel_token = input_tokens[i * seq_len];
        float first_pixel_value = (first_pixel_token / 127.5f) - 1.0f;
        unsigned char extracted_class = extract_class_from_first_pixel(first_pixel_value);
        
        char sample_filename[256];
        snprintf(sample_filename, sizeof(sample_filename), "sample_embedded_class_%d_idx_%d.png", extracted_class, i);
        save_mnist_image_png(&mnist_images[i * seq_len], sample_filename);
        printf("Saved sample %d: embedded_class=%d, file=%s\n", i, extracted_class, sample_filename);
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
    const int num_epochs = 200;
    const float learning_rate = 0.00001f;
    const int num_batches = num_images / batch_size;

    // Allocate device memory for batch data
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_sim(sim, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_sim(sim, d_target_tokens);
            if(loss >= 6.0) raise(SIGINT);
            
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_sim(sim);
            backward_pass_sim(sim, d_input_tokens);
            
            // Update weights
            update_weights_sim(sim, learning_rate);
            
            // Print progress
            if (batch % 2 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            }
            
            // Generate sample images periodically
            if (batch > 0 && batch % 500 == 0) {
                printf("\n--- Generating sample image (epoch %d, batch %d) ---\n", epoch, batch);
                
                unsigned char target_class = (unsigned char)(rand() % 10);
                
                float* generated_image = (float*)malloc(seq_len * sizeof(float));
                generate_image(sim, generated_image, 0.8f, d_input_tokens, target_class);
                
                // Save generated image with target class
                char gen_filename[256];
                snprintf(gen_filename, sizeof(gen_filename), "generated_epoch_%d_batch_%d_class_%d.png", epoch, batch, target_class);
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
    
    // Cleanup
    free(mnist_images);
    free(mnist_labels);
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free_sim(sim);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}