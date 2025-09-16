#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include "../data.h"

int main(void) {
    srand(time(NULL));
    
    const char* mnist_images_path = "../train-images-idx3-ubyte";
    
    // Create output directory
    struct stat st;
    if (stat("sample_images", &st) == -1) {
        mkdir("sample_images", 0755);
    }
    
    // Load MNIST data
    float* images = NULL;
    int num_images = 0;
    
    load_mnist_data(&images, &num_images, mnist_images_path);
    if (!images) {
        printf("Error: Could not load MNIST data\n");
        return 1;
    }
    
    printf("Successfully loaded %d MNIST images\n", num_images);
    
    // Save 10 randomly sampled images as PNG
    const int num_samples = 10;
    
    for (int i = 0; i < num_samples; i++) {
        // Random sample from dataset
        int random_idx = rand() % num_images;
        
        // Get pointer to the image data (28x28 floats)
        float* image_data = &images[random_idx * 28 * 28];
        
        // Save as PNG
        char filename[256];
        snprintf(filename, sizeof(filename), "sample_images/mnist_sample_%02d.png", i);
        save_mnist_image_png(image_data, filename);
        
        printf("Saved sample %d (index %d): %s\n", i, random_idx, filename);
        
        // Print some pixel values for verification
        printf("  First 5 pixels: [%.3f, %.3f, %.3f, %.3f, %.3f]\n", 
               image_data[0], image_data[1], image_data[2], image_data[3], image_data[4]);
    }
    
    // Save a CSV with the first 100 images for inspection
    char csv_filename[64];
    time_t now = time(NULL);
    strftime(csv_filename, sizeof(csv_filename), "%Y%m%d_%H%M%S_mnist_sample.csv", localtime(&now));
    
    int samples_to_save = (num_images < 100) ? num_images : 100;
    save_data(images, samples_to_save, 28 * 28, csv_filename);
    
    printf("\nCompleted! Saved %d PNG samples and CSV with %d images.\n", num_samples, samples_to_save);
    
    // Cleanup
    free(images);
    
    return 0;
}