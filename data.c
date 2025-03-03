#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <ctype.h>
#include <unistd.h>
#include <jansson.h>
#include <sys/stat.h>
#include <errno.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define COCO_ANNOTATIONS_URL "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
#define COCO_IMAGES_URL "http://images.cocodataset.org/zips/val2017.zip"
#define TEMP_DIR "coco_temp"
#define IMAGES_DIR "coco_images"
#define OUTPUT_FILE "coco_data.txt"
#define ANNOTATIONS_ZIP "annotations.zip"
#define IMAGES_ZIP "images.zip"
#define MAX_IMAGES 100  // Limit to download for demonstration
#define PROGRESS_BAR_WIDTH 50

// Structure to keep track of memory buffer
struct MemoryStruct {
    char *memory;
    size_t size;
};

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

int main() {
    // Initialize curl globally
    curl_global_init(CURL_GLOBAL_ALL);
    
    // Create temporary directories
    if (!ensure_directory(TEMP_DIR)) return 1;
    if (!ensure_directory(IMAGES_DIR)) return 1;
    
    char annotations_path[256];
    char images_path[256];
    sprintf(annotations_path, "%s/%s", TEMP_DIR, ANNOTATIONS_ZIP);
    sprintf(images_path, "%s/%s", TEMP_DIR, IMAGES_ZIP);
    
    // Step 1: Download annotations
    printf("Step 1: Downloading COCO annotations...\n");
    if (!download_file(COCO_ANNOTATIONS_URL, annotations_path, "annotations")) {
        return 1;
    }
    
    // Step 2: Extract annotations
    printf("Step 2: Extracting annotations...\n");
    char unzip_cmd[512];
    sprintf(unzip_cmd, "unzip -q -o %s -d %s", annotations_path, TEMP_DIR);
    if (!execute_command(unzip_cmd)) {
        return 1;
    }
    
    // Step 3: Parse captions
    printf("Step 3: Parsing captions...\n");
    char captions_path[256];
    sprintf(captions_path, "%s/annotations/captions_val2017.json", TEMP_DIR);
    
    FILE *captions_file = fopen(captions_path, "r");
    if (!captions_file) {
        fprintf(stderr, "Failed to open captions file: %s\n", captions_path);
        return 1;
    }
    
    // Read the entire JSON file into memory
    fseek(captions_file, 0, SEEK_END);
    long file_size = ftell(captions_file);
    rewind(captions_file);
    
    char *json_data = malloc(file_size + 1);
    if (!json_data) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(captions_file);
        return 1;
    }
    
    size_t bytes_read = fread(json_data, 1, file_size, captions_file);
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Error reading captions file\n");
        free(json_data);
        fclose(captions_file);
        return 1;
    }
    json_data[file_size] = '\0';
    fclose(captions_file);
    
    // Parse JSON data
    json_error_t error;
    json_t *root = json_loads(json_data, 0, &error);
    free(json_data);
    
    if (!root) {
        fprintf(stderr, "JSON parse error: %s\n", error.text);
        return 1;
    }
    
    // Get the annotations array
    json_t *annotations = json_object_get(root, "annotations");
    json_t *images = json_object_get(root, "images");
    
    if (!json_is_array(annotations) || !json_is_array(images)) {
        fprintf(stderr, "Invalid JSON structure\n");
        json_decref(root);
        return 1;
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
        
        // Limit for demonstration
        if (i >= MAX_IMAGES) break;
    }
    
    // Step 4: Download a subset of images from COCO
    printf("Step 4: Downloading COCO images (subset)...\n");
    if (!download_file(COCO_IMAGES_URL, images_path, "images")) {
        json_decref(image_map);
        json_decref(root);
        return 1;
    }
    
    // Step 5: Extract images
    printf("Step 5: Extracting images (this might take a while)...\n");
    sprintf(unzip_cmd, "unzip -q -o %s -d %s", images_path, IMAGES_DIR);
    if (!execute_command(unzip_cmd)) {
        json_decref(image_map);
        json_decref(root);
        return 1;
    }
    
    // Step 6: Create output file with captions
    printf("Step 6: Creating output file with image paths and captions...\n");
    FILE *output = fopen(OUTPUT_FILE, "w");
    if (!output) {
        fprintf(stderr, "Failed to open output file: %s\n", OUTPUT_FILE);
        json_decref(image_map);
        json_decref(root);
        return 1;
    }
    
    size_t annotation_count = json_array_size(annotations);
    size_t processed = 0;
    
    json_t *annotation;
    json_array_foreach(annotations, i, annotation) {
        json_t *image_id_json = json_object_get(annotation, "image_id");
        json_t *caption_json = json_object_get(annotation, "caption");
        
        if (json_is_integer(image_id_json) && json_is_string(caption_json)) {
            char id_str[32];
            sprintf(id_str, "%lld", json_integer_value(image_id_json));
            
            json_t *filename_json = json_object_get(image_map, id_str);
            if (filename_json && json_is_string(filename_json)) {
                fprintf(output, "%s/%s\t%s\n",
                        IMAGES_DIR,
                        json_string_value(filename_json),
                        json_string_value(caption_json));
                
                processed++;
                
                if (processed % 100 == 0) {
                    print_progress_bar(processed, MIN(annotation_count, MAX_IMAGES), "Processing");
                }
            }
        }
        
        // Limit for demonstration
        if (processed >= MAX_IMAGES) break;
    }
    
    print_progress_bar(processed, processed, "Processing");
    printf("\n");
    
    fclose(output);
    json_decref(image_map);
    json_decref(root);
    
    printf("\nProcessing complete:\n");
    printf("- Total images processed: %zu\n", processed);
    printf("- Output written to %s\n", OUTPUT_FILE);
    printf("- Format: [image_path]\\t[caption]\n");
    printf("\nYou can now use this data to train your image synthesis model.\n");
    
    curl_global_cleanup();
    return 0;
}