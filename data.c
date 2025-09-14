#include "data.h"

const char* CIFAR10_CLASS_NAMES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Load a single CIFAR-10 binary batch file
CIFAR10_Dataset* load_cifar10_batch(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open CIFAR-10 batch file: %s\n", filename);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Validate file size
    long expected_size = CIFAR10_IMAGES_PER_BATCH * CIFAR10_RECORD_SIZE;
    if (file_size != expected_size) {
        printf("Error: Invalid CIFAR-10 batch file size. Expected %ld, got %ld\n", 
               expected_size, file_size);
        fclose(file);
        return NULL;
    }

    // Allocate dataset
    CIFAR10_Dataset* dataset = (CIFAR10_Dataset*)malloc(sizeof(CIFAR10_Dataset));
    dataset->images = (CIFAR10_Image*)malloc(CIFAR10_IMAGES_PER_BATCH * sizeof(CIFAR10_Image));
    dataset->num_images = CIFAR10_IMAGES_PER_BATCH;
    dataset->capacity = CIFAR10_IMAGES_PER_BATCH;

    // Read all images
    for (int i = 0; i < CIFAR10_IMAGES_PER_BATCH; i++) {
        // Read label (1 byte)
        if (fread(&dataset->images[i].label, 1, 1, file) != 1) {
            printf("Error reading label for image %d\n", i);
            free_cifar10_dataset(dataset);
            fclose(file);
            return NULL;
        }

        // Read pixel data (3072 bytes)
        if (fread(dataset->images[i].pixels, 1, CIFAR10_TOTAL_PIXELS, file) != CIFAR10_TOTAL_PIXELS) {
            printf("Error reading pixel data for image %d\n", i);
            free_cifar10_dataset(dataset);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    printf("Loaded CIFAR-10 batch: %s (%d images)\n", filename, CIFAR10_IMAGES_PER_BATCH);
    return dataset;
}

// Load the entire CIFAR-10 dataset from extracted directory
CIFAR10_Dataset* load_cifar10_dataset(const char* data_dir) {
    // First, try to load training batches
    CIFAR10_Dataset* full_dataset = (CIFAR10_Dataset*)malloc(sizeof(CIFAR10_Dataset));
    full_dataset->images = NULL;
    full_dataset->num_images = 0;
    full_dataset->capacity = 0;

    char batch_filename[256];
    for (int batch = 1; batch <= 5; batch++) {
        snprintf(batch_filename, sizeof(batch_filename), "%s/data_batch_%d.bin", data_dir, batch);
        
        CIFAR10_Dataset* batch_data = load_cifar10_batch(batch_filename);
        if (!batch_data) {
            printf("Warning: Could not load batch %d\n", batch);
            continue;
        }

        // Expand full dataset capacity if needed
        int new_total = full_dataset->num_images + batch_data->num_images;
        if (new_total > full_dataset->capacity) {
            full_dataset->capacity = new_total;
            full_dataset->images = (CIFAR10_Image*)realloc(full_dataset->images, 
                                                           full_dataset->capacity * sizeof(CIFAR10_Image));
        }

        // Copy batch data to full dataset
        memcpy(&full_dataset->images[full_dataset->num_images], 
               batch_data->images, 
               batch_data->num_images * sizeof(CIFAR10_Image));
        full_dataset->num_images += batch_data->num_images;

        free_cifar10_dataset(batch_data);
    }

    if (full_dataset->num_images == 0) {
        printf("Error: No training batches loaded\n");
        free_cifar10_dataset(full_dataset);
        return NULL;
    }

    printf("Loaded full CIFAR-10 dataset: %d images\n", full_dataset->num_images);
    return full_dataset;
}

// Free dataset memory
void free_cifar10_dataset(CIFAR10_Dataset* dataset) {
    if (dataset) {
        if (dataset->images) {
            free(dataset->images);
        }
        free(dataset);
    }
}

// Convert CIFAR-10 format (R[1024], G[1024], B[1024]) to interleaved RGB
void convert_cifar10_to_rgb(const CIFAR10_Image* cifar_img, uint8_t* rgb_output) {
    for (int y = 0; y < CIFAR10_IMAGE_SIZE; y++) {
        for (int x = 0; x < CIFAR10_IMAGE_SIZE; x++) {
            int pixel_idx = y * CIFAR10_IMAGE_SIZE + x;
            int rgb_idx = pixel_idx * 3;
            
            // CIFAR-10 stores: R[1024], G[1024], B[1024]
            rgb_output[rgb_idx + 0] = cifar_img->pixels[pixel_idx];                    // R
            rgb_output[rgb_idx + 1] = cifar_img->pixels[pixel_idx + CIFAR10_PIXELS];   // G
            rgb_output[rgb_idx + 2] = cifar_img->pixels[pixel_idx + 2 * CIFAR10_PIXELS]; // B
        }
    }
}

// Save CIFAR-10 image as PNG
int save_cifar10_image_png(const CIFAR10_Image* image, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Could not create PNG file: %s\n", filename);
        return 0;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return 0;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);

    // Set PNG header
    png_set_IHDR(png, info, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    // Convert CIFAR-10 format to RGB
    uint8_t* rgb_data = (uint8_t*)malloc(CIFAR10_TOTAL_PIXELS * sizeof(uint8_t));
    convert_cifar10_to_rgb(image, rgb_data);

    // Write image data row by row
    png_bytep* row_pointers = (png_bytep*)malloc(CIFAR10_IMAGE_SIZE * sizeof(png_bytep));
    for (int y = 0; y < CIFAR10_IMAGE_SIZE; y++) {
        row_pointers[y] = &rgb_data[y * CIFAR10_IMAGE_SIZE * 3];
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    // Cleanup
    free(row_pointers);
    free(rgb_data);
    png_destroy_write_struct(&png, &info);
    fclose(fp);

    return 1;
}

// Print dataset statistics
void print_cifar10_stats(const CIFAR10_Dataset* dataset) {
    if (!dataset || !dataset->images) {
        printf("Dataset is empty or null\n");
        return;
    }

    int class_counts[10] = {0};
    
    // Count images per class
    for (int i = 0; i < dataset->num_images; i++) {
        if (dataset->images[i].label < 10) {
            class_counts[dataset->images[i].label]++;
        }
    }

    printf("\nCIFAR-10 Dataset Statistics:\n");
    printf("Total images: %d\n", dataset->num_images);
    printf("Image size: %dx%d pixels, %d channels\n", 
           CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, CIFAR10_CHANNELS);
    printf("\nClass distribution:\n");
    
    for (int i = 0; i < 10; i++) {
        printf("  %d (%s): %d images\n", i, CIFAR10_CLASS_NAMES[i], class_counts[i]);
    }
    printf("\n");
}