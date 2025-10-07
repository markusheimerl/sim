#include "data.h"

static uint32_t read_big_endian_uint32(FILE* file) {
    uint32_t val;
    fread(&val, sizeof(uint32_t), 1, file);
    return ((val & 0xFF) << 24) | (((val >> 8) & 0xFF) << 16) | 
           (((val >> 16) & 0xFF) << 8) | ((val >> 24) & 0xFF);
}

void load_mnist_data(unsigned char** X, int* num_samples, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open MNIST file: %s\n", filename);
        *X = NULL;
        *num_samples = 0;
        return;
    }
    
    uint32_t magic, num_imgs, rows, cols;
    magic = read_big_endian_uint32(file);
    num_imgs = read_big_endian_uint32(file);
    rows = read_big_endian_uint32(file);
    cols = read_big_endian_uint32(file);
    
    if (magic != 0x00000803 || rows != 28 || cols != 28) {
        printf("Error: Invalid MNIST file format\n");
        fclose(file);
        *X = NULL;
        *num_samples = 0;
        return;
    }
    
    *X = (unsigned char*)malloc(num_imgs * 28 * 28 * sizeof(unsigned char));
    fread(*X, sizeof(unsigned char), num_imgs * 28 * 28, file);
    
    fclose(file);
    
    *num_samples = (int)num_imgs;
    printf("Loaded MNIST data: %d samples (28x28)\n", *num_samples);
}

void load_mnist_labels(unsigned char** labels, int* num_labels, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open MNIST labels file: %s\n", filename);
        *labels = NULL;
        *num_labels = 0;
        return;
    }
    
    uint32_t magic, num_items;
    magic = read_big_endian_uint32(file);
    num_items = read_big_endian_uint32(file);
    
    if (magic != 0x00000801) {
        printf("Error: Invalid MNIST labels file format\n");
        fclose(file);
        *labels = NULL;
        *num_labels = 0;
        return;
    }
    
    *labels = (unsigned char*)malloc(num_items * sizeof(unsigned char));
    fread(*labels, sizeof(unsigned char), num_items, file);
    
    fclose(file);
    
    *num_labels = (int)num_items;
    printf("Loaded MNIST labels: %d labels\n", *num_labels);
}

void load_cifar10_data(unsigned char** X, unsigned char** labels, int* num_samples, const char** batch_files, int num_batches) {
    const int samples_per_batch = 10000;
    const int image_size = 32 * 32 * 3;  // 32x32 RGB
    
    *num_samples = samples_per_batch * num_batches;
    *X = (unsigned char*)malloc(*num_samples * image_size * sizeof(unsigned char));
    *labels = (unsigned char*)malloc(*num_samples * sizeof(unsigned char));
    
    int sample_offset = 0;
    
    for (int b = 0; b < num_batches; b++) {
        FILE* file = fopen(batch_files[b], "rb");
        if (!file) {
            printf("Error: Could not open CIFAR-10 file: %s\n", batch_files[b]);
            free(*X);
            free(*labels);
            *X = NULL;
            *labels = NULL;
            *num_samples = 0;
            return;
        }
        
        // Each record: 1 byte label + 3072 bytes image (1024 R + 1024 G + 1024 B)
        for (int i = 0; i < samples_per_batch; i++) {
            unsigned char label;
            unsigned char image[3072];
            
            fread(&label, sizeof(unsigned char), 1, file);
            fread(image, sizeof(unsigned char), 3072, file);
            
            (*labels)[sample_offset + i] = label;
            
            // CIFAR-10 format: R channel (1024) + G channel (1024) + B channel (1024)
            // We'll interleave them to RGB format for easier processing
            for (int pixel = 0; pixel < 1024; pixel++) {
                int out_idx = (sample_offset + i) * image_size + pixel * 3;
                (*X)[out_idx + 0] = image[pixel];           // R
                (*X)[out_idx + 1] = image[pixel + 1024];    // G
                (*X)[out_idx + 2] = image[pixel + 2048];    // B
            }
        }
        
        fclose(file);
        sample_offset += samples_per_batch;
        printf("Loaded CIFAR-10 batch: %s (%d samples)\n", batch_files[b], samples_per_batch);
    }
    
    printf("Total CIFAR-10 data loaded: %d samples (32x32x3)\n", *num_samples);
}

void save_data(unsigned char* X, int num_samples, int input_dim, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) { 
        printf("Error: cannot write %s\n", filename); 
        return; 
    }
    
    // Header
    for (int i = 0; i < input_dim; i++) {
        fprintf(f, "pixel%d%s", i, i == input_dim-1 ? "\n" : ",");
    }
    
    // Data
    for (int s = 0; s < num_samples; s++) {
        for (int i = 0; i < input_dim; i++) {
            fprintf(f, "%d%s", X[s * input_dim + i], i == input_dim-1 ? "\n" : ",");
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}

void save_mnist_image_png(unsigned char* image_data, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, 28, 28, 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(28 * sizeof(png_bytep));
    for (int y = 0; y < 28; y++) {
        row_pointers[y] = &image_data[y * 28];
    }

    png_write_rows(png, row_pointers, 28);
    png_write_end(png, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void save_cifar10_image_png(unsigned char* image_data, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, 32, 32, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(32 * sizeof(png_bytep));
    for (int y = 0; y < 32; y++) {
        row_pointers[y] = &image_data[y * 32 * 3];
    }

    png_write_rows(png, row_pointers, 32);
    png_write_end(png, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}