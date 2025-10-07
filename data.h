#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <png.h>

void load_mnist_data(unsigned char** X, int* num_samples, const char* filename);
void load_mnist_labels(unsigned char** labels, int* num_labels, const char* filename);
void load_cifar10_data(unsigned char** X, unsigned char** labels, int* num_samples, const char** batch_files, int num_batches);
void save_data(unsigned char* X, int num_samples, int input_dim, const char* filename);
void save_mnist_image_png(unsigned char* image_data, const char* filename);
void save_cifar10_image_png(unsigned char* image_data, const char* filename);

#endif