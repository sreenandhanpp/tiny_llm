#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} Tensor;

Tensor tensor_create(size_t rows, size_t cols);
void tensor_free(Tensor *tensor);
void tensor_fill(Tensor *tensor, float value);
void tensor_print(const Tensor *tensor);

#endif