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

// Function to save a color image
void save_color_image(const char *filename, unsigned char *data, int width, int height, int channels) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPARRAY buffer;
    int row_stride = width * channels;
    
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    cinfo.in_color_space = (channels == 3) ? JCS_RGB : JCS_GRAYSCALE;
    
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    while (cinfo.next_scanline < cinfo.image_height) {
        memcpy(buffer[0], data + cinfo.next_scanline * row_stride, row_stride);
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

// Function to shift Hartley transform
void hartley_shift(double *data, double *shifted, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate destination coordinates after shift
            int dst_y = (y + height/2) % height;
            int dst_x = (x + width/2) % width;
            
            // Copy data with shift
            shifted[dst_y * width + dst_x] = data[y * width + x];
        }
    }
}

// Function to inverse shift Hartley transform
void hartley_ishift(double *shifted, double *data, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate source coordinates before shift
            int src_y = (y + height/2) % height;
            int src_x = (x + width/2) % width;
            
            // Copy data with inverse shift
            data[y * width + x] = shifted[src_y * width + src_x];
        }
    }
}

// Function to apply low-pass filter to shifted Hartley transform
void apply_low_pass_filter(double *shifted, int width, int height, double cutoff_radius) {
    int center_x = width / 2;
    int center_y = height / 2;
    double radius_squared = cutoff_radius * cutoff_radius;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate distance from center (squared)
            double dx = x - center_x;
            double dy = y - center_y;
            double distance_squared = dx*dx + dy*dy;
            
            // Apply filter (hard cutoff)
            if (distance_squared > radius_squared) {
                shifted[y * width + x] = 0.0;
            }
        }
    }
}

// Function to save Hartley visualization
void save_visualization(const char *filename, double **data, int width, int height, int is_filtered) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPARRAY buffer;
    
    // Create a composite visualization using the magnitudes from all three channels
    double *vis_data = (double *)malloc(width * height * sizeof(double));
    memset(vis_data, 0, width * height * sizeof(double));
    
    // Sum magnitudes from all channels
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {  // RGB channels
            vis_data[i] += fabs(data[c][i]);
        }
        vis_data[i] /= 3.0;  // Average magnitude
    }
    
    // For visualization, we'll use logarithmic scaling
    double min_val = vis_data[0];
    double max_val = vis_data[0];
    for (int i = 1; i < width * height; i++) {
        if (vis_data[i] < min_val) min_val = vis_data[i];
        if (vis_data[i] > max_val) max_val = vis_data[i];
    }
    
    // Apply log scaling: log(1 + |x|)
    for (int i = 0; i < width * height; i++) {
        vis_data[i] = log(1.0 + fabs(vis_data[i]));
    }
    
    // Find new min/max after scaling
    min_val = vis_data[0];
    max_val = vis_data[0];
    for (int i = 1; i < width * height; i++) {
        if (vis_data[i] < min_val) min_val = vis_data[i];
        if (vis_data[i] > max_val) max_val = vis_data[i];
    }
    
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
    
    printf("Saved %s (Hartley transform %s visualization)\n", 
           filename, is_filtered ? "filtered" : "unfiltered");
}

// Function to compute the 2D Hartley Transform for a single channel
double* compute_hartley_transform(unsigned char *channel_data, int width, int height) {
    // Allocate memory for FFTW
    double *input = (double *)fftw_malloc(sizeof(double) * width * height);
    double *hartley = (double *)fftw_malloc(sizeof(double) * width * height);
    
    // Copy channel data to double array for processing
    for (int i = 0; i < width * height; i++) {
        input[i] = (double)channel_data[i];
    }
    
    // Use r2r (real-to-real) transform with DHT (discrete Hartley transform)
    fftw_plan plan = fftw_plan_r2r_2d(height, width, input, hartley, FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
    fftw_execute(plan);
    
    // FFTW computes an unnormalized transform, so normalize by sqrt(N)
    double norm_factor = 1.0 / sqrt(width * height);
    for (int i = 0; i < width * height; i++) {
        hartley[i] *= norm_factor;
    }
    
    // Clean up FFTW resources
    fftw_destroy_plan(plan);
    fftw_free(input);
    
    // Return the result
    return hartley;
}

// Function to compute the inverse 2D Hartley Transform for a single channel
void inverse_hartley_transform(double *hartley, unsigned char *channel_data, int width, int height) {
    // Allocate memory for FFTW
    double *output = (double *)fftw_malloc(sizeof(double) * width * height);
    
    // Create FFTW plan for inverse DHT
    // DHT is its own inverse, just need to scale afterward
    fftw_plan plan = fftw_plan_r2r_2d(height, width, hartley, output, FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
    fftw_execute(plan);
    
    // Normalize by 1/N for inverse transform
    double norm_factor = 1.0 / (width * height);
    
    // Convert back to unsigned char with proper scaling
    double min_val = output[0] * norm_factor;
    double max_val = min_val;
    
    // Find min/max values
    for (int i = 1; i < width * height; i++) {
        double val = output[i] * norm_factor;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    // Scale and convert to unsigned char
    for (int i = 0; i < width * height; i++) {
        // Normalize to 0-255 range if needed
        if (max_val > 255 || min_val < 0) {
            double normalized = (output[i] * norm_factor - min_val) / (max_val - min_val);
            channel_data[i] = (unsigned char)(normalized * 255.0);
        } else {
            // Just clip values
            double val = output[i] * norm_factor;
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            channel_data[i] = (unsigned char)val;
        }
    }
    
    // Clean up FFTW resources
    fftw_destroy_plan(plan);
    fftw_free(output);
}

// Function to extract a single channel from an RGB image
unsigned char* extract_channel(Image img, int channel_index) {
    unsigned char *channel_data = (unsigned char *)malloc(img.width * img.height);
    
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int src_idx = (y * img.width + x) * img.channels + channel_index;
            int dst_idx = y * img.width + x;
            channel_data[dst_idx] = img.data[src_idx];
        }
    }
    
    return channel_data;
}

// Function to combine color channels into an RGB image
unsigned char* combine_channels(unsigned char *red, unsigned char *green, unsigned char *blue, int width, int height) {
    unsigned char *rgb = (unsigned char *)malloc(width * height * 3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = y * width + x;
            int dst_idx = (y * width + x) * 3;
            
            rgb[dst_idx + 0] = red[src_idx];
            rgb[dst_idx + 1] = green[src_idx];
            rgb[dst_idx + 2] = blue[src_idx];
        }
    }
    
    return rgb;
}

int main() {
    // Load the image
    Image img = load_jpeg("img.jpg");
    if (img.data == NULL) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
    
    printf("Image loaded: %d x %d with %d channels\n", img.width, img.height, img.channels);
    
    // If image is grayscale, return error
    if (img.channels < 3) {
        fprintf(stderr, "This version expects a color image with at least 3 channels\n");
        free(img.data);
        return 1;
    }
    
    // Extract individual channels
    unsigned char *red_channel = extract_channel(img, 0);
    unsigned char *green_channel = extract_channel(img, 1);
    unsigned char *blue_channel = extract_channel(img, 2);
    
    // Compute the Hartley transform for each channel
    double *hartley_red = compute_hartley_transform(red_channel, img.width, img.height);
    double *hartley_green = compute_hartley_transform(green_channel, img.width, img.height);
    double *hartley_blue = compute_hartley_transform(blue_channel, img.width, img.height);
    
    // Create shifted versions for filtering
    double *shifted_red = (double *)fftw_malloc(sizeof(double) * img.width * img.height);
    double *shifted_green = (double *)fftw_malloc(sizeof(double) * img.width * img.height);
    double *shifted_blue = (double *)fftw_malloc(sizeof(double) * img.width * img.height);
    
    hartley_shift(hartley_red, shifted_red, img.width, img.height);
    hartley_shift(hartley_green, shifted_green, img.width, img.height);
    hartley_shift(hartley_blue, shifted_blue, img.width, img.height);
    
    // Save visualization of shifted (unfiltered) transforms
    double *shifted_transforms[3] = {shifted_red, shifted_green, shifted_blue};
    save_visualization("hartley.jpg", shifted_transforms, img.width, img.height, 0);
    
    // Apply low-pass filter to each channel
    double radius = img.width * 0.00495;  // 20% of image width
    printf("Applying low-pass filter with radius %.1f pixels\n", radius);
    
    apply_low_pass_filter(shifted_red, img.width, img.height, radius);
    apply_low_pass_filter(shifted_green, img.width, img.height, radius);
    apply_low_pass_filter(shifted_blue, img.width, img.height, radius);
    
    // Save visualization of filtered transforms
    save_visualization("hartley_filtered.jpg", shifted_transforms, img.width, img.height, 1);
    
    // Inverse shift the filtered data
    double *filtered_red = (double *)fftw_malloc(sizeof(double) * img.width * img.height);
    double *filtered_green = (double *)fftw_malloc(sizeof(double) * img.width * img.height);
    double *filtered_blue = (double *)fftw_malloc(sizeof(double) * img.width * img.height);
    
    hartley_ishift(shifted_red, filtered_red, img.width, img.height);
    hartley_ishift(shifted_green, filtered_green, img.width, img.height);
    hartley_ishift(shifted_blue, filtered_blue, img.width, img.height);
    
    // Compute inverse Hartley transform for each channel
    unsigned char *filtered_red_channel = (unsigned char *)malloc(img.width * img.height);
    unsigned char *filtered_green_channel = (unsigned char *)malloc(img.width * img.height);
    unsigned char *filtered_blue_channel = (unsigned char *)malloc(img.width * img.height);
    
    inverse_hartley_transform(filtered_red, filtered_red_channel, img.width, img.height);
    inverse_hartley_transform(filtered_green, filtered_green_channel, img.width, img.height);
    inverse_hartley_transform(filtered_blue, filtered_blue_channel, img.width, img.height);
    
    // Combine the filtered channels back into a color image
    unsigned char *filtered_color = combine_channels(
        filtered_red_channel, filtered_green_channel, filtered_blue_channel, 
        img.width, img.height
    );
    
    // Save the final filtered color image
    save_color_image("img_filtered.jpg", filtered_color, img.width, img.height, 3);
    printf("Filtered color image saved as 'img_filtered.jpg'\n");
    
    // Clean up
    free(img.data);
    free(red_channel);
    free(green_channel);
    free(blue_channel);
    free(filtered_red_channel);
    free(filtered_green_channel);
    free(filtered_blue_channel);
    free(filtered_color);
    
    fftw_free(hartley_red);
    fftw_free(hartley_green);
    fftw_free(hartley_blue);
    fftw_free(shifted_red);
    fftw_free(shifted_green);
    fftw_free(shifted_blue);
    fftw_free(filtered_red);
    fftw_free(filtered_green);
    fftw_free(filtered_blue);
    
    return 0;
}