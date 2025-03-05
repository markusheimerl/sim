#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include "ssm/gpu/ssm.h"

#define MAX_LINE_LENGTH 1000000

// ---------------------------------------------------------------------
// Function: Propagate gradients between stacked models
// ---------------------------------------------------------------------
void backward_between_models(SSM* first_model, SSM* second_model, float* d_first_model_input) {
    // Zero gradients for first model
    zero_gradients(first_model);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute gradient from state path: d_input_grad = B^T * state_error
    CHECK_CUBLAS(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           first_model->output_dim, first_model->batch_size, second_model->state_dim,
                           &alpha,
                           second_model->d_B, second_model->state_dim,
                           second_model->d_state_error, second_model->state_dim,
                           &beta,
                           first_model->d_error, first_model->output_dim));
    
    // Add gradient from direct path: d_input_grad += D^T * error
    CHECK_CUBLAS(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           first_model->output_dim, first_model->batch_size, second_model->output_dim,
                           &alpha,
                           second_model->d_D, second_model->output_dim,
                           second_model->d_error, second_model->output_dim,
                           &alpha, // Add to existing gradient
                           first_model->d_error, first_model->output_dim));
    
    // Now do the backward pass for the first model
    backward_pass(first_model, d_first_model_input);
}

// Count lines in a file
int count_lines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return -1;
    }
    
    int count = 0;
    char c;
    int prev_char = '\n';  // Start assuming we're at beginning of a line
    
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') {
            count++;
        }
        prev_char = c;
    }
    
    // If file doesn't end with newline, count the last line
    if (prev_char != '\n') {
        count++;
    }
    
    fclose(file);
    return count;
}

// Load and preprocess CSV data with batching
int load_csv_data(const char* filename, float** h_data_time_major, int* num_timesteps, int* batch_size, int* feature_dim) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return 0;
    }
    
    // Count lines to determine total rows
    int total_lines = count_lines(filename);
    if (total_lines <= 0) {
        fprintf(stderr, "Error counting lines or empty file\n");
        fclose(file);
        return 0;
    }
    
    // Reset file position
    rewind(file);
    
    // Read first line to determine input dimension (features per sample)
    char* line = (char*)malloc(MAX_LINE_LENGTH);
    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        fprintf(stderr, "Error reading first line\n");
        free(line);
        fclose(file);
        return 0;
    }
    
    // Count commas to determine dimension
    int dim = 1;
    for (char* c = line; *c != '\0' && *c != '\n'; c++) {
        if (*c == ',') dim++;
    }
    
    // Reset file position
    rewind(file);
    
    // Determine proper batch size and number of timesteps
    // Each image is processed as a batch sample
    // Each denoising step is a timestep
    
    // In the denoising dataset, we have NUM_NOISE_STEPS (1024) rows per image
    // and 100 images total
    int num_noise_steps = 1024;
    int num_images = total_lines / num_noise_steps;
    
    printf("Found %d lines with %d dimensions each\n", total_lines, dim);
    printf("Detected %d images, each with %d noise steps\n", num_images, num_noise_steps);
    
    *batch_size = num_images;     // Each image is a batch sample
    *num_timesteps = num_noise_steps;  // Each denoising step is a timestep
    *feature_dim = dim;          // Dimension of each sample
    
    // Allocate data arrays: [timesteps][batch_size][feature_dim]
    *h_data_time_major = (float*)malloc((*num_timesteps) * (*batch_size) * dim * sizeof(float));
    
    // Temporary array to store all data in row-major format (batch/timestep sequential in CSV)
    float* temp_data = (float*)malloc(total_lines * dim * sizeof(float));
    
    // Read all data from CSV
    for (int row = 0; row < total_lines; row++) {
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            break;
        }
        
        // Parse CSV line
        char* token = strtok(line, ",");
        for (int d = 0; d < dim && token != NULL; d++) {
            float value = atof(token);
            // Store in temporary array
            temp_data[row * dim + d] = value;
            token = strtok(NULL, ",");
        }
    }
    
    // Now reorganize data into time-major format
    // In the CSV, data is organized as:
    // [image 0, timestep 0]
    // [image 0, timestep 1]
    // ...
    // [image 0, timestep 1023]
    // [image 1, timestep 0]
    // ...

    // We want to reorganize as:
    // [image 0, timestep 0]
    // [image 1, timestep 0]
    // ...
    // [image 99, timestep 0]
    // [image 0, timestep 1]
    // [image 1, timestep 1]
    // ...
    
    for (int t = 0; t < *num_timesteps; t++) {
        for (int b = 0; b < *batch_size; b++) {
            // Source index in the original data
            int src_idx = (b * (*num_timesteps) + t) * dim;
            
            // Destination index in time-major format
            int dst_idx = (t * (*batch_size) + b) * dim;
            
            // Copy this sample's features
            memcpy((*h_data_time_major) + dst_idx, temp_data + src_idx, dim * sizeof(float));
        }
    }
    
    // Free temporary storage
    free(temp_data);
    free(line);
    fclose(file);
    
    return dim;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    const char* csv_file = "denoising_dataset_hilbert_1024_steps.csv";
    
    // Model parameters
    int input_dim = 0;       // Will be determined from CSV
    int state_dim = 1024;    // State dimension
    int layer1_dim = 128;    // Layer 1 hidden dimension
    int layer2_dim = 128;    // Layer 2 hidden dimension
    int layer3_dim = 128;    // Layer 3 hidden dimension
    float learning_rate = 0.0001; // Learning rate
    int num_epochs = 10;      // Number of training epochs
    
    printf("=== SIGM Denoising Model Training ===\n");
    printf("Loading denoising data from: %s\n", csv_file);
    
    // Load data from CSV
    float* h_data_time_major = NULL;
    int num_timesteps = 0;
    int batch_size = 0;
    int feature_dim = 0;
    
    input_dim = load_csv_data(csv_file, &h_data_time_major, &num_timesteps, &batch_size, &feature_dim);
    if (input_dim == 0) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }
    
    printf("Data loaded: %d images, %d denoising steps, %d dimensions\n", 
           batch_size, num_timesteps, input_dim);
    printf("Using state dimension: %d\n", state_dim);
    printf("Hidden layer dimensions: %d, %d, %d\n", layer1_dim, layer2_dim, layer3_dim);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("Training epochs: %d\n\n", num_epochs);
    
    // Transfer data to GPU
    float *d_data_time_major;
    CHECK_CUDA(cudaMalloc(&d_data_time_major, num_timesteps * batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data_time_major, h_data_time_major, 
                         num_timesteps * batch_size * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Free host memory as it's no longer needed
    free(h_data_time_major);
    
    // Initialize SSM models
    SSM* layer1_ssm;
    SSM* layer2_ssm;
    SSM* layer3_ssm;
    SSM* layer4_ssm;
    
    if (argc == 5) {
        // Load models from files
        layer1_ssm = load_ssm(argv[1], batch_size);
        layer2_ssm = load_ssm(argv[2], batch_size);
        layer3_ssm = load_ssm(argv[3], batch_size);
        layer4_ssm = load_ssm(argv[4], batch_size);
        printf("Successfully loaded pretrained models\n");
    } else {
        // Initialize from scratch
        layer1_ssm = init_ssm(input_dim, state_dim, layer1_dim, batch_size);
        layer2_ssm = init_ssm(layer1_dim, state_dim, layer2_dim, batch_size);
        layer3_ssm = init_ssm(layer2_dim, state_dim, layer3_dim, batch_size);
        layer4_ssm = init_ssm(layer3_dim, state_dim, input_dim, batch_size);
        printf("Initialized new models\n");
    }
    
    // Calculate and print total parameter count
    long long total_params = 0;
    
    // SSM parameters for each layer
    long long layer1_params = (long long)state_dim * state_dim +  // A matrix
                             state_dim * input_dim +              // B matrix
                             layer1_dim * state_dim +             // C matrix
                             layer1_dim * input_dim;              // D matrix
    
    long long layer2_params = (long long)state_dim * state_dim +  // A matrix
                             state_dim * layer1_dim +             // B matrix
                             layer2_dim * state_dim +             // C matrix
                             layer2_dim * layer1_dim;             // D matrix
    
    long long layer3_params = (long long)state_dim * state_dim +  // A matrix
                             state_dim * layer2_dim +             // B matrix
                             layer3_dim * state_dim +             // C matrix
                             layer3_dim * layer2_dim;             // D matrix
    
    long long layer4_params = (long long)state_dim * state_dim +  // A matrix
                             state_dim * layer3_dim +             // B matrix
                             input_dim * state_dim +              // C matrix
                             input_dim * layer3_dim;              // D matrix
    
    total_params = layer1_params + layer2_params + layer3_params + layer4_params;
    
    printf("Model parameter count:\n");
    printf("  Layer 1 SSM: %lld parameters\n", layer1_params);
    printf("  Layer 2 SSM: %lld parameters\n", layer2_params);
    printf("  Layer 3 SSM: %lld parameters\n", layer3_params);
    printf("  Layer 4 SSM: %lld parameters\n", layer4_params);
    printf("  Total:       %lld parameters (%.2f million)\n", total_params, total_params / 1000000.0);
    
    // Allocate memory for intermediate outputs
    float* d_layer1_output;
    float* d_layer2_output;
    float* d_layer3_output;
    
    CHECK_CUDA(cudaMalloc(&d_layer1_output, batch_size * layer1_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_output, batch_size * layer2_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_output, batch_size * layer3_dim * sizeof(float)));
    
    printf("\nStarting training for %d epochs with %d steps...\n", num_epochs, num_timesteps);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(layer1_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer2_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer3_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer4_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        
        float epoch_loss = 0.0f;
        
        // Process each timestep (denoising step)
        for (int t = 0; t < num_timesteps - 1; t++) {
            // Get current timestep inputs and targets
            // For denoising, input is current noise level, target is next noise level (less noise)
            float* d_X_t = d_data_time_major + t * batch_size * input_dim;
            float* d_y_t = d_data_time_major + (t + 1) * batch_size * input_dim;
            
            // Forward pass: layer 1 SSM
            forward_pass(layer1_ssm, d_X_t);
            
            // Copy layer1 output for layer2 input
            CHECK_CUDA(cudaMemcpy(d_layer1_output, layer1_ssm->d_predictions, 
                              batch_size * layer1_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 2 SSM
            forward_pass(layer2_ssm, d_layer1_output);
            
            // Copy layer2 output for layer3 input
            CHECK_CUDA(cudaMemcpy(d_layer2_output, layer2_ssm->d_predictions, 
                              batch_size * layer2_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 3 SSM
            forward_pass(layer3_ssm, d_layer2_output);
            
            // Copy layer3 output for layer4 input
            CHECK_CUDA(cudaMemcpy(d_layer3_output, layer3_ssm->d_predictions, 
                              batch_size * layer3_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 4 SSM (output layer)
            forward_pass(layer4_ssm, d_layer3_output);
            
            // Calculate MSE loss
            float loss = calculate_loss(layer4_ssm, d_y_t);
            epoch_loss += loss;
            
            // Backward pass: layer 4 SSM (output layer)
            zero_gradients(layer4_ssm);
            backward_pass(layer4_ssm, d_layer3_output);
            
            // Backward pass: layer 3 SSM
            backward_between_models(layer3_ssm, layer4_ssm, d_layer2_output);
            
            // Backward pass: layer 2 SSM
            backward_between_models(layer2_ssm, layer3_ssm, d_layer1_output);
            
            // Backward pass: layer 1 SSM
            backward_between_models(layer1_ssm, layer2_ssm, d_X_t);
            
            // Update weights
            update_weights(layer1_ssm, learning_rate);
            update_weights(layer2_ssm, learning_rate);
            update_weights(layer3_ssm, learning_rate);
            update_weights(layer4_ssm, learning_rate);
            
            // Print progress
            if (t == 0 || t == num_timesteps - 2 || (t + 1) % 100 == 0) {
                printf("Epoch %d/%d, Step %d/%d, Loss: %f, Avg Loss: %f\n", 
                       epoch + 1, num_epochs, t + 1, num_timesteps - 1, 
                       loss, epoch_loss/(t+1));
            }
        }
        
        // Calculate average epoch loss
        float avg_epoch_loss = epoch_loss / (num_timesteps - 1);
        printf("Epoch %d/%d completed, Average Loss: %f\n", epoch + 1, num_epochs, avg_epoch_loss);
    }
    
    // Save the final models
    char model_time[20];
    time_t now = time(NULL);
    struct tm *timeinfo = localtime(&now);
    strftime(model_time, sizeof(model_time), "%Y%m%d_%H%M%S", timeinfo);
    
    char layer1_fname[64], layer2_fname[64], layer3_fname[64], layer4_fname[64];
    sprintf(layer1_fname, "%s_layer1.bin", model_time);
    sprintf(layer2_fname, "%s_layer2.bin", model_time);
    sprintf(layer3_fname, "%s_layer3.bin", model_time);
    sprintf(layer4_fname, "%s_layer4.bin", model_time);
    
    save_ssm(layer1_ssm, layer1_fname);
    save_ssm(layer2_ssm, layer2_fname);
    save_ssm(layer3_ssm, layer3_fname);
    save_ssm(layer4_ssm, layer4_fname);
    
    printf("Models saved with prefix: %s\n", model_time);
    
    // Clean up
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
    
    cudaFree(d_data_time_major);
    cudaFree(d_layer1_output);
    cudaFree(d_layer2_output);
    cudaFree(d_layer3_output);
    
    printf("\nTraining completed!\n");
    return 0;
}