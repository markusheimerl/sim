#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <png.h>

#define CIFAR10_IMAGE_SIZE 32
#define CIFAR10_CHANNELS 3
#define CIFAR10_PIXELS (CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE)
#define CIFAR10_TOTAL_PIXELS (CIFAR10_PIXELS * CIFAR10_CHANNELS)
#define CIFAR10_RECORD_SIZE (1 + CIFAR10_TOTAL_PIXELS)
#define CIFAR10_IMAGES_PER_BATCH 10000

typedef struct {
    uint8_t label;
    uint8_t pixels[CIFAR10_TOTAL_PIXELS];
} CIFAR10_Image;

typedef struct {
    CIFAR10_Image* images;
    int num_images;
    int capacity;
} CIFAR10_Dataset;

// Function prototypes
CIFAR10_Dataset* load_cifar10_batch(const char* filename);
void free_cifar10_dataset(CIFAR10_Dataset* dataset);
void convert_cifar10_to_rgb(const CIFAR10_Image* cifar_img, uint8_t* rgb_output);
int save_cifar10_image_png(const CIFAR10_Image* image, const char* filename);
void print_cifar10_stats(const CIFAR10_Dataset* dataset);

extern const char* CIFAR10_CLASS_NAMES[10];

#endif