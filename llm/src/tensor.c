#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

Tensor tensor_create(size_t rows, size_t cols)
{
    Tensor tensor = {NULL, rows, cols};

    if (rows == 0 || cols == 0 ||
        rows > (size_t)-1 / cols ||
        rows * cols > (size_t)-1 / sizeof(float)) {
        tensor.rows = 0;
        tensor.cols = 0;
        return tensor;
    }

    tensor.data = calloc(rows * cols, sizeof(float));

    if (tensor.data == NULL) {
        tensor.rows = 0;
        tensor.cols = 0;
    }

    return tensor;
}

void tensor_free(Tensor *tensor)
{
    if (tensor == NULL) {
        return;
    }

    free(tensor->data);
    tensor->data = NULL;
    tensor->rows = 0;
    tensor->cols = 0;
}

void tensor_fill(Tensor *tensor, float value)
{
    if (tensor == NULL || tensor->data == NULL) {
        return;
    }

    for (size_t i = 0; i < tensor->rows * tensor->cols; i++) {
        tensor->data[i] = value;
    }
}

void tensor_print(const Tensor *tensor)
{
    if (tensor == NULL || tensor->data == NULL) {
        return;
    }

    for (size_t i = 0; i < tensor->rows; i++) {
        for (size_t j = 0; j < tensor->cols; j++) {
            printf("%8.3f ", tensor->data[i * tensor->cols + j]);
        }
        printf("\n");
    }
}