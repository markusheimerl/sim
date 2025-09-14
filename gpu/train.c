#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include "../data.h"

int extract_cifar10_if_needed(const char* tar_path, const char* extract_dir) {
    struct stat st = {0};
    
    // Check if extraction directory already exists
    if (stat(extract_dir, &st) == 0) {
        printf("CIFAR-10 data already extracted in %s\n", extract_dir);
        return 1;
    }
    
    // Create extraction directory
    if (mkdir(extract_dir, 0755) != 0) {
        printf("Error: Could not create extraction directory: %s\n", extract_dir);
        return 0;
    }
    
    // Extract tar.gz file
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

int main() {
    srand(time(NULL));
    
    // Paths
    const char* tar_path = "../cifar-10-binary.tar.gz";
    const char* extract_dir = "../cifar-10-data";
    const char* output_dir = "sample_images";
    
    // Check if tar file exists
    if (access(tar_path, F_OK) != 0) {
        printf("Error: CIFAR-10 tar file not found: %s\n", tar_path);
        printf("Please download cifar-10-binary.tar.gz from https://www.cs.toronto.edu/~kriz/cifar.html\n");
        return 1;
    }
    
    // Extract CIFAR-10 data if needed
    if (!extract_cifar10_if_needed(tar_path, extract_dir)) {
        return 1;
    }
    
    // Create output directory for sample images
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0755);
    }
    
    // Load first training batch for testing
    char batch_path[256];
    snprintf(batch_path, sizeof(batch_path), "%s/data_batch_1.bin", extract_dir);
    
    CIFAR10_Dataset* dataset = load_cifar10_batch(batch_path);
    if (!dataset) {
        printf("Error: Could not load CIFAR-10 training batch\n");
        return 1;
    }
    
    // Print dataset statistics
    print_cifar10_stats(dataset);
    
    // Save some sample images
    printf("Saving sample images to %s/\n", output_dir);
    
    // Save one image from each class (first occurrence)
    int class_saved[10] = {0};
    int samples_saved = 0;
    
    for (int i = 0; i < dataset->num_images && samples_saved < 10; i++) {
        uint8_t label = dataset->images[i].label;
        
        if (label < 10 && !class_saved[label]) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/sample_%d_%s.png", 
                    output_dir, label, CIFAR10_CLASS_NAMES[label]);
            
            if (save_cifar10_image_png(&dataset->images[i], filename)) {
                printf("  Saved: %s (image %d)\n", filename, i);
                class_saved[label] = 1;
                samples_saved++;
            } else {
                printf("  Failed to save: %s\n", filename);
            }
        }
    }
    
    // Save a few random samples
    printf("\nSaving additional random samples...\n");
    for (int i = 0; i < 5; i++) {
        int random_idx = rand() % dataset->num_images;
        CIFAR10_Image* img = &dataset->images[random_idx];
        
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/random_%d_label_%d_%s.png", 
                output_dir, i, img->label, CIFAR10_CLASS_NAMES[img->label]);
        
        if (save_cifar10_image_png(img, filename)) {
            printf("  Saved: %s (image %d)\n", filename, random_idx);
        } else {
            printf("  Failed to save: %s\n", filename);
        }
    }
    
    printf("\nImage rendering complete! Check the %s/ directory.\n", output_dir);
    
    // Cleanup
    free_cifar10_dataset(dataset);
    
    return 0;
}