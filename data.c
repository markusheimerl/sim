#include "data.h"

const char* CIFAR10_CLASS_NAMES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

CIFAR10_Dataset* load_cifar10_batch(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open CIFAR-10 batch file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    long expected_size = CIFAR10_IMAGES_PER_BATCH * CIFAR10_RECORD_SIZE;
    if (file_size != expected_size) {
        printf("Error: Invalid CIFAR-10 batch file size. Expected %ld, got %ld\n", 
               expected_size, file_size);
        fclose(file);
        return NULL;
    }

    CIFAR10_Dataset* dataset = (CIFAR10_Dataset*)malloc(sizeof(CIFAR10_Dataset));
    dataset->images = (CIFAR10_Image*)malloc(CIFAR10_IMAGES_PER_BATCH * sizeof(CIFAR10_Image));
    dataset->num_images = CIFAR10_IMAGES_PER_BATCH;
    dataset->capacity = CIFAR10_IMAGES_PER_BATCH;

    for (int i = 0; i < CIFAR10_IMAGES_PER_BATCH; i++) {
        if (fread(&dataset->images[i].label, 1, 1, file) != 1) {
            printf("Error reading label for image %d\n", i);
            free_cifar10_dataset(dataset);
            fclose(file);
            return NULL;
        }

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

void free_cifar10_dataset(CIFAR10_Dataset* dataset) {
    if (dataset) {
        if (dataset->images) {
            free(dataset->images);
        }
        free(dataset);
    }
}

void convert_cifar10_to_rgb(const CIFAR10_Image* cifar_img, uint8_t* rgb_output) {
    for (int y = 0; y < CIFAR10_IMAGE_SIZE; y++) {
        for (int x = 0; x < CIFAR10_IMAGE_SIZE; x++) {
            int pixel_idx = y * CIFAR10_IMAGE_SIZE + x;
            int rgb_idx = pixel_idx * 3;
            
            rgb_output[rgb_idx + 0] = cifar_img->pixels[pixel_idx];
            rgb_output[rgb_idx + 1] = cifar_img->pixels[pixel_idx + CIFAR10_PIXELS];
            rgb_output[rgb_idx + 2] = cifar_img->pixels[pixel_idx + 2 * CIFAR10_PIXELS];
        }
    }
}

int save_cifar10_image_png(const CIFAR10_Image* image, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return 0;

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
    png_set_IHDR(png, info, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    uint8_t* rgb_data = (uint8_t*)malloc(CIFAR10_TOTAL_PIXELS * sizeof(uint8_t));
    convert_cifar10_to_rgb(image, rgb_data);

    png_bytep* row_pointers = (png_bytep*)malloc(CIFAR10_IMAGE_SIZE * sizeof(png_bytep));
    for (int y = 0; y < CIFAR10_IMAGE_SIZE; y++) {
        row_pointers[y] = &rgb_data[y * CIFAR10_IMAGE_SIZE * 3];
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    free(row_pointers);
    free(rgb_data);
    png_destroy_write_struct(&png, &info);
    fclose(fp);

    return 1;
}

void print_cifar10_stats(const CIFAR10_Dataset* dataset) {
    if (!dataset || !dataset->images) {
        printf("Dataset is empty or null\n");
        return;
    }

    int class_counts[10] = {0};
    
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