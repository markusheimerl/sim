#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jpeglib.h>
#include <fftw3.h>

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

// Function to save a grayscale image
void save_grayscale_image(const char *filename, unsigned char *data, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPARRAY buffer;
    
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
    
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width, 1);
    
    while (cinfo.next_scanline < cinfo.image_height) {
        memcpy(buffer[0], data + cinfo.next_scanline * width, width);
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

// Function to save 2D data as a visualization image
void save_data_as_image(const char *filename, double *data, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPARRAY buffer;
    
    // Create a copy of the data for visualization
    double *vis_data = (double *)malloc(width * height * sizeof(double));
    memcpy(vis_data, data, width * height * sizeof(double));
    
    // For visualization, we'll use logarithmic scaling with proper DC handling
    // First, shift everything to the positive domain by adding the minimum value
    double min_val = vis_data[0];
    double max_val = vis_data[0];
    for (int i = 1; i < width * height; i++) {
        if (vis_data[i] < min_val) min_val = vis_data[i];
        if (vis_data[i] > max_val) max_val = vis_data[i];
    }
    
    printf("Initial value range - Min: %f, Max: %f\n", min_val, max_val);
    
    // Apply log scaling: log(1 + |x|), preserving signs
    for (int i = 0; i < width * height; i++) {
        double sign = (vis_data[i] >= 0) ? 1.0 : -1.0;
        vis_data[i] = sign * log(1.0 + fabs(vis_data[i]));
    }
    
    // Find new min/max after scaling
    min_val = vis_data[0];
    max_val = vis_data[0];
    for (int i = 1; i < width * height; i++) {
        if (vis_data[i] < min_val) min_val = vis_data[i];
        if (vis_data[i] > max_val) max_val = vis_data[i];
    }
    
    printf("After log scaling - Min: %f, Max: %f\n", min_val, max_val);
    
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        free(vis_data);
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
    
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width, 1);
    
    while (cinfo.next_scanline < cinfo.image_height) {
        for (int i = 0; i < width; i++) {
            // Normalize to 0-255 range
            double normalized = (vis_data[cinfo.next_scanline * width + i] - min_val) / (max_val - min_val);
            buffer[0][i] = (JSAMPLE)(normalized * 255.0);
        }
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    free(vis_data);
}

// Function to compute the 2D Hartley Transform correctly
double* compute_hartley_transform(unsigned char *grayscale_image, int width, int height) {
    // Allocate memory for FFTW
    double *input = (double *)fftw_malloc(sizeof(double) * width * height);
    double *hartley = (double *)fftw_malloc(sizeof(double) * width * height);
    
    // Copy grayscale image to double array for processing
    for (int i = 0; i < width * height; i++) {
        input[i] = (double)grayscale_image[i];
    }
    
    // Use r2r (real-to-real) transform with DHT (discrete Hartley transform)
    fftw_plan plan = fftw_plan_r2r_2d(height, width, input, hartley, FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
    fftw_execute(plan);
    
    
    // Clean up FFTW resources
    fftw_destroy_plan(plan);
    fftw_free(input);
    
    // Return the result
    return hartley;
}

// Convert color image to grayscale
unsigned char* convert_to_grayscale(Image img) {
    unsigned char *grayscale = (unsigned char *)malloc(img.width * img.height);
    
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = (y * img.width + x) * img.channels;
            
            if (img.channels >= 3) {
                // Standard luminance formula
                grayscale[y * img.width + x] = (unsigned char)(
                    0.299 * img.data[idx] +      // Red
                    0.587 * img.data[idx + 1] +  // Green
                    0.114 * img.data[idx + 2]);  // Blue
            } else {
                // Already grayscale
                grayscale[y * img.width + x] = img.data[idx];
            }
        }
    }
    
    return grayscale;
}

int main() {
    // Load the image
    Image img = load_jpeg("img.jpg");
    if (img.data == NULL) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
    
    printf("Image loaded: %d x %d with %d channels\n", img.width, img.height, img.channels);
    
    // Convert to grayscale
    unsigned char *grayscale = convert_to_grayscale(img);
    
    // Save the grayscale image so we can see what goes into the transform
    save_grayscale_image("grayscale_input.jpg", grayscale, img.width, img.height);
    printf("Grayscale image saved\n");
    
    // Compute the Hartley transform using FFTW's DHT implementation
    double *hartley_data = compute_hartley_transform(grayscale, img.width, img.height);
    
    // Save visualization of the Hartley transform
    save_data_as_image("hartley_transform.jpg", hartley_data, img.width, img.height);
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
    free(grayscale);
    fftw_free(hartley_data);
    
    return 0;
}