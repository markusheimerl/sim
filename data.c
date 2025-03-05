#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curl/curl.h>
#include <ctype.h>
#include <unistd.h>
#include <jansson.h>
#include <jpeglib.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// URLs and directory paths
#define COCO_ANNOTATIONS_URL "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
#define COCO_IMAGES_URL "http://images.cocodataset.org/zips/val2017.zip"
#define TEMP_DIR "coco_temp"
#define IMAGES_DIR "coco_images"
#define ANNOTATIONS_ZIP "annotations.zip"
#define IMAGES_ZIP "images.zip"
#define MAX_IMAGES 64
#define PROGRESS_BAR_WIDTH 50

// Denoising parameters
#define NUM_NOISE_STEPS 1024
#define MAX_DIMENSION 128

// Image structure
typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// Structure to keep track of memory buffer
struct MemoryStruct {
    char *memory;
    size_t size;
};

//=====================================================================
// CURL and Download Functions
//=====================================================================

// Write callback for curl (file)
size_t write_file_callback(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Progress callback for curl
int progress_callback(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    const char *label = (const char*)clientp;
    (void)ultotal;  // Mark as unused
    (void)ulnow;    // Mark as unused
    
    if (dltotal <= 0) return 0; // Avoid division by zero
    
    // Calculate percentage
    double percentage = (double)dlnow / (double)dltotal;
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    // Print progress bar
    printf("\rDownloading %s: [", label);
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        if (i < filled_width) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% (%.2f/%.2f MB)", 
           percentage * 100, 
           (double)dlnow / 1048576, // Convert to MB
           (double)dltotal / 1048576);
    
    fflush(stdout);
    return 0; // Return 0 to continue transfer
}

// Function to print a progress bar
void print_progress_bar(long current, long total, const char* phase) {
    double percentage = (double)current / (double)total;
    if (percentage > 1.0) percentage = 1.0; // Cap at 100%
    
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    printf("\r%s: [", phase);
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        if (i < filled_width) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% (%ld/%ld)", percentage * 100, current, total);
    fflush(stdout);
}

// Function to create directory if it doesn't exist
int ensure_directory(const char *dir) {
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0700) != 0) {
            fprintf(stderr, "Failed to create directory %s: %s\n", dir, strerror(errno));
            return 0;
        }
    }
    return 1;
}

// Function to download a file with progress bar
int download_file(const char *url, const char *filepath, const char *label) {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    
    curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Failed to initialize curl\n");
        return 0;
    }
    
    fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filepath);
        curl_easy_cleanup(curl);
        return 0;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, (void *)label);
    
    res = curl_easy_perform(curl);
    fclose(fp);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "\nDownload failed: %s\n", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        return 0;
    }
    
    printf("\nDownload complete.\n");
    curl_easy_cleanup(curl);
    return 1;
}

// Execute a system command and check for errors
int execute_command(const char *command) {
    printf("Executing: %s\n", command);
    int result = system(command);
    if (result != 0) {
        fprintf(stderr, "Command failed with exit code %d\n", result);
        return 0;
    }
    return 1;
}

//=====================================================================
// Image Processing Functions
//=====================================================================

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

// Scale an image to a new size using nearest neighbor interpolation
Image scale_image(Image src, int new_width, int new_height) {
    Image dst = {0};
    dst.width = new_width;
    dst.height = new_height;
    dst.channels = src.channels;
    dst.data = (unsigned char *)malloc(new_width * new_height * src.channels);
    
    double x_ratio = (double)src.width / new_width;
    double y_ratio = (double)src.height / new_height;
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);
            
            // Clamp to image bounds
            if (src_x >= src.width) src_x = src.width - 1;
            if (src_y >= src.height) src_y = src.height - 1;
            
            for (int c = 0; c < src.channels; c++) {
                int src_idx = (src_y * src.width + src_x) * src.channels + c;
                int dst_idx = (y * new_width + x) * src.channels + c;
                dst.data[dst_idx] = src.data[src_idx];
            }
        }
    }
    
    return dst;
}

// Calculate scaling factors to resize an image to a maximum dimension
void calculate_scaling_factors(int width, int height, int max_dim, int *new_width, int *new_height) {
    if (width <= max_dim && height <= max_dim) {
        // Image is already small enough
        *new_width = width;
        *new_height = height;
    } else if (width > height) {
        // Width is the larger dimension
        *new_width = max_dim;
        *new_height = (int)(height * ((double)max_dim / width));
    } else {
        // Height is the larger dimension
        *new_height = max_dim;
        *new_width = (int)(width * ((double)max_dim / height));
    }
}

// Extract center square from an image
Image extract_center_square(Image img) {
    Image square = {0};
    
    // Determine the size of the square (minimum of width and height)
    int size = (img.width < img.height) ? img.width : img.height;
    
    // Calculate top-left corner of the square
    int start_x = (img.width - size) / 2;
    int start_y = (img.height - size) / 2;
    
    // Create the square image
    square.width = size;
    square.height = size;
    square.channels = img.channels;
    square.data = (unsigned char *)malloc(size * size * img.channels);
    
    // Copy the center square from the original image
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            for (int c = 0; c < img.channels; c++) {
                int src_idx = ((start_y + y) * img.width + (start_x + x)) * img.channels + c;
                int dst_idx = (y * size + x) * img.channels + c;
                square.data[dst_idx] = img.data[src_idx];
            }
        }
    }
    
    return square;
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

//=====================================================================
// Hilbert Curve and Noise Functions
//=====================================================================

// Rotate/flip a quadrant appropriately for Hilbert curve
void rotate(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }
        // Swap x and y
        int t = *x;
        *x = *y;
        *y = t;
    }
}

// Convert (x,y) to d (distance along Hilbert curve)
int xy2d(int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rotate(s, &x, &y, rx, ry);
    }
    return d;
}

// Find the next power of 2
int next_power_of_2(int n) {
    int power = 1;
    while (power < n) power *= 2;
    return power;
}

// Structure to hold 2D coordinates and Hilbert indices
typedef struct {
    int x, y;      // 2D coordinates
    int hilbert_d; // Hilbert distance
} HilbertPoint;

// Comparison function for qsort
int compare_hilbert(const void *a, const void *b) {
    return ((HilbertPoint*)a)->hilbert_d - ((HilbertPoint*)b)->hilbert_d;
}

// Create a mapping from 2D to 1D using Hilbert curve ordering
void create_hilbert_mapping(int width, int height, int **to_1d, int **to_2d) {
    // Determine n (power of 2 >= max(width, height))
    int n = next_power_of_2((width > height) ? width : height);
    
    // Allocate memory for map arrays
    *to_1d = (int*)malloc(width * height * sizeof(int));
    *to_2d = (int*)malloc(width * height * sizeof(int));
    
    // Create array of points with their Hilbert distances
    HilbertPoint *points = (HilbertPoint*)malloc(width * height * sizeof(HilbertPoint));
    int idx = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            points[idx].x = x;
            points[idx].y = y;
            points[idx].hilbert_d = xy2d(n, x, y);
            idx++;
        }
    }
    
    // Sort points by Hilbert distance
    qsort(points, width * height, sizeof(HilbertPoint), compare_hilbert);
    
    // Fill in the mapping arrays
    for (int i = 0; i < width * height; i++) {
        int orig_idx = points[i].y * width + points[i].x;
        (*to_1d)[orig_idx] = i;  // Maps from 2D to 1D
        (*to_2d)[i] = orig_idx;  // Maps from 1D to 2D
    }
    
    free(points);
}

// Flatten a grayscale image using Hilbert curve ordering
float* flatten_with_hilbert_float(unsigned char *grayscale, int width, int height, int **to_2d) {
    // Create forward and backward mappings
    int *to_1d;
    create_hilbert_mapping(width, height, &to_1d, to_2d);
    
    // Create the flattened array as float (normalized 0-1)
    float *flattened = (float*)malloc(width * height * sizeof(float));
    
    // Flatten using the mapping and normalize to 0-1
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int orig_idx = y * width + x;
            int flat_idx = to_1d[orig_idx];
            flattened[flat_idx] = grayscale[orig_idx] / 255.0f;
        }
    }
    
    free(to_1d);
    return flattened;
}

// Generate pure noise image with values between 0-1
void generate_pure_noise(float *data, int length) {
    for (int i = 0; i < length; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Add Gaussian noise with specified standard deviation to a normalized float array
void add_gaussian_noise_float(float *data, int length, float noise_level) {
    for (int i = 0; i < length; i++) {
        double u1 = (double)rand() / (float)RAND_MAX;
        double u2 = (double)rand() / (float)RAND_MAX;
        
        // Avoid log(0)
        if (u1 < 1e-8) u1 = 1e-8;
        
        // Box-Muller transform
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        
        // Add noise to normalized data
        data[i] += (float)(z0 * noise_level);
        
        // Clamp values between 0 and 1
        if (data[i] < 0.0f) data[i] = 0.0f;
        if (data[i] > 1.0f) data[i] = 1.0f;
    }
}

// Blend between source image and target image with given factor (0.0 = source, 1.0 = target)
void blend_images(float *result, float *source, float *target, int length, float blend_factor) {
    for (int i = 0; i < length; i++) {
        result[i] = source[i] * (1.0f - blend_factor) + target[i] * blend_factor;
    }
}

// Unflatten a 1D array back to a 2D image
unsigned char* unflatten_with_hilbert(float *flattened, int width, int height, int *to_2d) {
    unsigned char *unflattened = (unsigned char*)malloc(width * height);
    
    for (int i = 0; i < width * height; i++) {
        // Convert back from normalized float to byte
        unflattened[to_2d[i]] = (unsigned char)(flattened[i] * 255.0f);
    }
    
    return unflattened;
}

// Write a float array to a CSV file as a row
void write_float_array_to_csv(FILE *fp, float *array, int length) {
    for (int i = 0; i < length; i++) {
        fprintf(fp, "%.6f", array[i]);
        if (i < length - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");
}

// Check if a file has a specific extension
int has_extension(const char *filename, const char *ext) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return 0;
    return strcasecmp(dot + 1, ext) == 0;
}

//=====================================================================
// Functions for Image Download and Caption Processing
//=====================================================================

// Function to sanitize caption for use as a filename
void sanitize_filename(char *filename, const char *caption) {
    int i, j = 0;
    
    // Copy and sanitize the caption to use as a filename
    for (i = 0; caption[i] != '\0' && j < 80; i++) {
        char c = caption[i];
        // Replace spaces with underscores and remove special characters
        if (isalnum(c) || c == '_' || c == '-' || c == ' ') {
            if (c == ' ') c = '_';
            filename[j++] = c;
        }
    }
    
    // Ensure we don't exceed filename length limits and add extension
    filename[MIN(j, 80)] = '\0';  // Truncate if too long
    strcat(filename, ".jpg");
}

// Download COCO images with caption-based filenames
int download_coco_images() {
    // Create temporary directories
    if (!ensure_directory(TEMP_DIR)) return 0;
    if (!ensure_directory(IMAGES_DIR)) return 0;
    
    char annotations_path[256];
    char images_path[256];
    sprintf(annotations_path, "%s/%s", TEMP_DIR, ANNOTATIONS_ZIP);
    sprintf(images_path, "%s/%s", TEMP_DIR, IMAGES_ZIP);
    
    // Step 1: Download annotations
    printf("Step 1: Downloading COCO annotations...\n");
    if (!download_file(COCO_ANNOTATIONS_URL, annotations_path, "annotations")) {
        return 0;
    }
    
    // Step 2: Extract annotations
    printf("Step 2: Extracting annotations...\n");
    char unzip_cmd[512];
    sprintf(unzip_cmd, "unzip -q -o %s -d %s", annotations_path, TEMP_DIR);
    if (!execute_command(unzip_cmd)) {
        return 0;
    }
    
    // Step 3: Parse captions
    printf("Step 3: Parsing captions...\n");
    char captions_path[256];
    sprintf(captions_path, "%s/annotations/captions_val2017.json", TEMP_DIR);
    
    FILE *captions_file = fopen(captions_path, "r");
    if (!captions_file) {
        fprintf(stderr, "Failed to open captions file: %s\n", captions_path);
        return 0;
    }
    
    // Read the entire JSON file into memory
    fseek(captions_file, 0, SEEK_END);
    long file_size = ftell(captions_file);
    rewind(captions_file);
    
    char *json_data = malloc(file_size + 1);
    if (!json_data) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(captions_file);
        return 0;
    }
    
    size_t bytes_read = fread(json_data, 1, file_size, captions_file);
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Error reading captions file\n");
        free(json_data);
        fclose(captions_file);
        return 0;
    }
    json_data[file_size] = '\0';
    fclose(captions_file);
    
    // Parse JSON data
    json_error_t error;
    json_t *root = json_loads(json_data, 0, &error);
    free(json_data);
    
    if (!root) {
        fprintf(stderr, "JSON parse error: %s\n", error.text);
        return 0;
    }
    
    // Get the annotations array
    json_t *annotations = json_object_get(root, "annotations");
    json_t *images = json_object_get(root, "images");
    
    if (!json_is_array(annotations) || !json_is_array(images)) {
        fprintf(stderr, "Invalid JSON structure\n");
        json_decref(root);
        return 0;
    }
    
    // Create a map of image_id to file_name
    size_t image_count = json_array_size(images);
    printf("Found %zu images in metadata\n", image_count);
    
    // Map from image_id to filename
    json_t *image_map = json_object();
    
    size_t i;
    json_t *image;
    json_array_foreach(images, i, image) {
        json_t *id_json = json_object_get(image, "id");
        json_t *file_name_json = json_object_get(image, "file_name");
        
        if (json_is_integer(id_json) && json_is_string(file_name_json)) {
            char id_str[32];
            sprintf(id_str, "%lld", json_integer_value(id_json));
            json_object_set(image_map, id_str, file_name_json);
        }
    }
    
    // Step 4: Download images
    printf("Step 4: Downloading COCO images (subset)...\n");
    if (!download_file(COCO_IMAGES_URL, images_path, "images")) {
        json_decref(image_map);
        json_decref(root);
        return 0;
    }
    
    // Step 5: Extract images
    printf("Step 5: Extracting images (this might take a while)...\n");
    sprintf(unzip_cmd, "unzip -q -o %s -d %s", images_path, TEMP_DIR);
    if (!execute_command(unzip_cmd)) {
        json_decref(image_map);
        json_decref(root);
        return 0;
    }
    
    // Step 6: Process images and create caption-based filenames
    printf("Step 6: Processing images with caption-based filenames...\n");
    
    size_t annotation_count = json_array_size(annotations);
    size_t processed = 0;
    
    // Track which image IDs we've already processed to avoid duplicates
    json_t *processed_images = json_object();
    
    json_t *annotation;
    json_array_foreach(annotations, i, annotation) {
        json_t *image_id_json = json_object_get(annotation, "image_id");
        json_t *caption_json = json_object_get(annotation, "caption");
        
        if (json_is_integer(image_id_json) && json_is_string(caption_json)) {
            long long image_id = json_integer_value(image_id_json);
            const char *caption = json_string_value(caption_json);
            
            char id_str[32];
            sprintf(id_str, "%lld", image_id);
            
            // Check if we've already processed this image (to avoid duplicates)
            if (json_object_get(processed_images, id_str) != NULL) {
                continue;
            }
            
            json_t *filename_json = json_object_get(image_map, id_str);
            if (filename_json && json_is_string(filename_json)) {
                const char *original_filename = json_string_value(filename_json);
                
                // Generate sanitized filename from caption
                char sanitized_filename[100];
                sanitize_filename(sanitized_filename, caption);
                
                // Create complete paths
                char src_path[512], dest_path[512];
                sprintf(src_path, "%s/val2017/%s", TEMP_DIR, original_filename);
                sprintf(dest_path, "%s/%s", IMAGES_DIR, sanitized_filename);
                
                // Copy the file to destination with new name
                char copy_cmd[1024];
                sprintf(copy_cmd, "cp %s %s", src_path, dest_path);
                if (execute_command(copy_cmd)) {
                    // Mark this image as processed
                    json_object_set(processed_images, id_str, json_true());
                    
                    processed++;
                    
                    if (processed % 10 == 0) {
                        print_progress_bar(processed, MIN(annotation_count, MAX_IMAGES), "Processing");
                    }
                }
            }
        }
        
        // Limit for demonstration
        if (processed >= MAX_IMAGES) break;
    }
    
    print_progress_bar(processed, processed, "Processing");
    printf("\n");
    
    json_decref(processed_images);
    json_decref(image_map);
    json_decref(root);
    
    // Step 7: Clean up temporary directory
    printf("Step 7: Cleaning up temporary files...\n");
    char cleanup_cmd[256];
    sprintf(cleanup_cmd, "rm -rf %s", TEMP_DIR);
    if (!execute_command(cleanup_cmd)) {
        fprintf(stderr, "Warning: Failed to remove temporary directory %s\n", TEMP_DIR);
    }
    
    printf("\nImage processing complete:\n");
    printf("- Total images processed: %zu\n", processed);
    printf("- Images saved to %s/ directory with caption filenames\n", IMAGES_DIR);
    printf("- Temporary files removed from %s\n", TEMP_DIR);
    
    return processed > 0;
}

//=====================================================================
// Functions for Creating Denoising Dataset
//=====================================================================

// Process a single image and add its noise sequences to the CSV file
// Returns 1 if successful, 0 otherwise
int process_image_for_denoising(const char *filepath, FILE *csv_file, int is_first_image, int *square_size) {
    // Load the image
    Image img = load_jpeg(filepath);
    if (img.data == NULL) {
        fprintf(stderr, "Failed to load image: %s\n", filepath);
        return 0;
    }
    
    // Step 1: Scale down the image
    int new_width, new_height;
    calculate_scaling_factors(img.width, img.height, MAX_DIMENSION, &new_width, &new_height);
    
    Image scaled_img;
    if (new_width != img.width || new_height != img.height) {
        scaled_img = scale_image(img, new_width, new_height);
        free(img.data); // Free the original image data
    } else {
        scaled_img = img; // Just use the original
    }
    
    // Step 2: Extract center square from the scaled image
    Image square_img = extract_center_square(scaled_img);
    free(scaled_img.data); // Free the scaled image data
    
    // Return the square size for the caller
    *square_size = square_img.width;
    
    // Step 3: Convert to grayscale
    unsigned char *grayscale = convert_to_grayscale(square_img);
    
    // Step 4: Create Hilbert mapping and flatten the image
    int *to_2d;
    float *clean_flattened = flatten_with_hilbert_float(grayscale, square_img.width, square_img.height, &to_2d);
    
    // Get the length of the flattened array
    int flattened_length = square_img.width * square_img.height;
    
    // Allocate an array to hold all the noisy versions
    float **noisy_arrays = (float**)malloc(NUM_NOISE_STEPS * sizeof(float*));
    
    // Create a pure noise image for the most noisy step
    float *pure_noise = (float*)malloc(flattened_length * sizeof(float));
    generate_pure_noise(pure_noise, flattened_length);
    
    // Create increasingly noisy versions of the image
    printf("Generating %d noise steps for %s...\n", NUM_NOISE_STEPS, filepath);
    
    // For the first image only, save visualization of noise stages
    int *vis_steps = NULL;
    int num_vis_steps = 0;
    /*
    if (is_first_image) {
        // Define visualization steps for the first image only
        int steps[] = {0, 1, 4, 16, 64, 256, 512, 768, 1022, 1023};
        num_vis_steps = sizeof(steps) / sizeof(steps[0]);
        vis_steps = malloc(num_vis_steps * sizeof(int));
        memcpy(vis_steps, steps, num_vis_steps * sizeof(int));
    }
    */
    for (int i = 0; i < NUM_NOISE_STEPS; i++) {
        // Allocate memory for this noise step
        noisy_arrays[i] = (float*)malloc(flattened_length * sizeof(float));
        
        // Calculate the blend factor (0 = clean image, 1 = pure noise)
        float blend_factor = (float)i / (NUM_NOISE_STEPS - 1);
        
        // Apply a power curve to make higher noise levels even more noisy
        blend_factor = powf(blend_factor, 0.5f);
        
        // For the last few steps, ensure we reach pure noise
        if (i >= NUM_NOISE_STEPS - 5) {
            // Blend quickly to pure noise in the last 5 steps
            float last_steps_factor = (float)(i - (NUM_NOISE_STEPS - 5)) / 4.0f;
            blend_factor = blend_factor * (1.0f - last_steps_factor) + last_steps_factor;
        }
        
        // Use the blend factor to interpolate between clean image and pure noise
        blend_images(noisy_arrays[i], clean_flattened, pure_noise, flattened_length, blend_factor);
        
        // Add additional random noise to avoid too smooth transitions
        float noise_level = 0.05f * blend_factor; // Scale noise with blend factor
        add_gaussian_noise_float(noisy_arrays[i], flattened_length, noise_level);
        
        // Make the final step pure noise
        if (i == NUM_NOISE_STEPS - 1) {
            memcpy(noisy_arrays[i], pure_noise, flattened_length * sizeof(float));
        }
        
        // For the first image only, visualize select noise levels
        if (is_first_image && vis_steps != NULL) {
            for (int v = 0; v < num_vis_steps; v++) {
                if (i == vis_steps[v]) {
                    // Convert back to image and save for visualization
                    unsigned char *noisy_image = unflatten_with_hilbert(noisy_arrays[i], 
                                                                       square_img.width, square_img.height, to_2d);
                    char vis_filename[256];
                    sprintf(vis_filename, "noisy_%d_of_%d.jpg", i, NUM_NOISE_STEPS-1);
                    save_grayscale_image(vis_filename, noisy_image, square_img.width, square_img.height);
                    free(noisy_image);
                    
                    printf("Saved visualization for noise step %d (blend factor: %.4f)\n", 
                           i, blend_factor);
                }
            }
        }
        
        // Show progress sparingly to avoid console spam
        if (i == 0 || i == NUM_NOISE_STEPS-1 || i % (NUM_NOISE_STEPS/10) == 0) {
            printf("  Generated noise step %d/%d (blend factor: %.4f)\n", 
                   i+1, NUM_NOISE_STEPS, blend_factor);
        }
    }
    
    // Write all noise steps to the CSV file in reverse order (most noisy to clean)
    for (int i = NUM_NOISE_STEPS - 1; i >= 0; i--) {
        write_float_array_to_csv(csv_file, noisy_arrays[i], flattened_length);
    }
    
    // Clean up all noisy arrays
    for (int i = 0; i < NUM_NOISE_STEPS; i++) {
        free(noisy_arrays[i]);
    }
    free(noisy_arrays);
    free(pure_noise);
    
    // Clean up other allocations
    free(square_img.data);
    free(grayscale);
    free(clean_flattened);
    free(to_2d);
    
    // Free visualization steps array if it was allocated
    if (vis_steps) {
        free(vis_steps);
    }
    
    return 1;
}

// Create denoising dataset from images in directory
int create_denoising_dataset() {
    // Open directory
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir(IMAGES_DIR);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", IMAGES_DIR);
        return 0;
    }
    
    // Count JPEG files in directory
    int jpg_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (has_extension(entry->d_name, "jpg") || has_extension(entry->d_name, "jpeg")) {
            jpg_count++;
            if (jpg_count >= MAX_IMAGES) {
                break;
            }
        }
    }
    rewinddir(dir);
    
    if (jpg_count == 0) {
        fprintf(stderr, "No JPEG images found in %s\n", IMAGES_DIR);
        closedir(dir);
        return 0;
    }
    
    printf("Found %d JPEG images in %s\n", jpg_count, IMAGES_DIR);
    
    // Create CSV file
    char csv_filename[256];
    sprintf(csv_filename, "denoising_dataset_hilbert_%d_steps.csv", NUM_NOISE_STEPS);
    FILE *csv_file = fopen(csv_filename, "w");
    if (!csv_file) {
        fprintf(stderr, "Failed to create CSV file %s\n", csv_filename);
        closedir(dir);
        return 0;
    }
    
    // Process each JPEG file
    int processed_count = 0;
    int square_size = 0;
    
    while ((entry = readdir(dir)) != NULL && processed_count < MAX_IMAGES) {
        if (has_extension(entry->d_name, "jpg") || has_extension(entry->d_name, "jpeg")) {
            char filepath[512];
            sprintf(filepath, "%s/%s", IMAGES_DIR, entry->d_name);
            
            printf("\nProcessing image %d/%d: %s\n", processed_count + 1, jpg_count, entry->d_name);
            
            // Process the image and add its data to the CSV
            int is_first_image = (processed_count == 0);
            if (process_image_for_denoising(filepath, csv_file, is_first_image, &square_size)) {
                processed_count++;
            }
        }
    }
    
    closedir(dir);
    fclose(csv_file);
    
    printf("\nDenoising dataset creation complete!\n");
    printf("- Processed %d images.\n", processed_count);
    printf("- Dataset saved to %s\n", csv_filename);
    printf("- Square image dimensions: %d x %d\n", square_size, square_size);
    printf("- Each image contributed %d noise sequences\n", NUM_NOISE_STEPS);
    printf("- Total dataset size: %d rows x %d columns\n", 
           processed_count * NUM_NOISE_STEPS, square_size * square_size);
    
    return 1;
}

//=====================================================================
// Main Program
//=====================================================================

int main() {
    // Initialize random seed for noise generation
    srand(time(NULL));
    
    // Initialize curl for HTTP requests
    curl_global_init(CURL_GLOBAL_ALL);
    
    printf("=== COCO Image and Denoising Dataset Generator ===\n\n");
    printf("This program will:\n");
    printf("1. Download COCO images with caption-based filenames\n");
    printf("2. Create a denoising dataset from these images in Hilbert space\n\n");
    
    // Phase 1: Download and process COCO images
    printf("=== PHASE 1: Downloading and Processing COCO Images ===\n\n");
    if (!download_coco_images()) {
        fprintf(stderr, "Failed to download and process COCO images\n");
        curl_global_cleanup();
        return 1;
    }
    
    // Phase 2: Create denoising dataset
    printf("\n=== PHASE 2: Creating Denoising Dataset ===\n\n");
    if (!create_denoising_dataset()) {
        fprintf(stderr, "Failed to create denoising dataset\n");
        curl_global_cleanup();
        return 1;
    }
    
    // Clean up curl
    curl_global_cleanup();
    
    printf("\n=== All Processing Complete! ===\n");
    printf("You can now use this dataset to train your denoising model.\n");
    
    return 0;
}