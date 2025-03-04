#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <fftw3.h>  // We'll use FFTW3 to implement the Hartley transform

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// Function to load a JPEG image
Image load_jpeg(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;
    Image img = {0};
    
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return img;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    
    img.width = cinfo.output_width;
    img.height = cinfo.output_height;
    img.channels = cinfo.output_components;
    img.data = (unsigned char *)malloc(img.width * img.height * img.channels);
    
    row_stride = img.width * img.channels;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(img.data + (cinfo.output_scanline - 1) * row_stride, buffer[0], row_stride);
    }
    
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return img;
}

// Function to save 2D data as a grayscale image for visualization
void save_data_as_image(const char *filename, double *data, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPARRAY buffer;
    int row_stride;
    
    // Find min and max for normalization
    double min_val = data[0];
    double max_val = data[0];
    for (int i = 1; i < width * height; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;
    
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    
    row_stride = width;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    while (cinfo.next_scanline < cinfo.image_height) {
        for (int i = 0; i < width; i++) {
            double normalized = (data[cinfo.next_scanline * width + i] - min_val) / (max_val - min_val);
            buffer[0][i] = (JSAMPLE)(normalized * 255.0);
        }
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

// Function to compute the 2D Hartley Transform using FFTW
double* compute_hartley_transform(unsigned char *image_data, int width, int height) {
    // Allocate memory for input and output
    double *input = (double *)fftw_malloc(sizeof(double) * width * height);
    double *output = (double *)fftw_malloc(sizeof(double) * width * height);
    fftw_complex *fft_result = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height);
    
    // Convert image to grayscale and copy to input buffer
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            input[idx] = (double)image_data[idx * 3]; // Just use the red channel for simplicity
        }
    }
    
    // Create FFTW plan for forward FFT
    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(height, width, input, fft_result, FFTW_ESTIMATE);
    
    // Execute FFT
    fftw_execute(forward_plan);
    
    // Compute Hartley transform from FFT result
    // H(u,v) = Re{F(u,v)} - Im{F(u,v)}
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int fft_idx = y * width + x;
            
            if (y >= height/2 || x >= width/2) {
                // Handle conjugate symmetry for second half
                int y_sym = (y > 0) ? height - y : 0;
                int x_sym = (x > 0) ? width - x : 0;
                int fft_idx_sym = y_sym * width + x_sym;
                
                // For points outside the stored half, use conjugate symmetry
                output[idx] = fft_result[fft_idx_sym][0] - fft_result[fft_idx_sym][1];
            } else {
                output[idx] = fft_result[fft_idx][0] - fft_result[fft_idx][1];
            }
        }
    }
    
    // Clean up FFTW resources
    fftw_destroy_plan(forward_plan);
    fftw_free(input);
    fftw_free(fft_result);
    
    // Return the result
    return output;
}

void save_hartley_data(const char *filename, double *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open file %s for writing\n", filename);
        return;
    }
    
    // Write dimensions
    fwrite(&width, sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    
    // Write data
    fwrite(data, sizeof(double), width * height, f);
    
    fclose(f);
}

int main() {
    // Load the bear image
    Image img = load_jpeg("bear.jpg");
    if (img.data == NULL) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
    
    printf("Image loaded: %d x %d with %d channels\n", img.width, img.height, img.channels);
    
    // Compute the Hartley transform
    double *hartley_data = compute_hartley_transform(img.data, img.width, img.height);
    
    // Save visualization of the Hartley transform
    save_data_as_image("hartley_transform.jpg", hartley_data, img.width, img.height);
    
    // Save raw hartley transform data
    save_hartley_data("hartley_transform.bin", hartley_data, img.width, img.height);
    
    printf("Hartley transform computed and saved\n");
    
    // Print a small sample of the transform (e.g., top-left corner)
    printf("Sample of Hartley Transform (top-left 5x5):\n");
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("%10.2f ", hartley_data[y * img.width + x]);
        }
        printf("\n");
    }
    
    // Clean up
    free(img.data);
    fftw_free(hartley_data);
    
    return 0;
}