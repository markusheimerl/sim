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