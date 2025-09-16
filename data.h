#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <png.h>

void load_mnist_data(float** X, int* num_samples, const char* filename);
void save_data(float* X, int num_samples, int input_dim, const char* filename);
void save_mnist_image_png(float* image_data, const char* filename);

#endif