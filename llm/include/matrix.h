#ifndef MATRIX_H
#define MATRIX_H

/*
 * matrix.h
 *
 * Declares the matrix and vector operations used by the
 * Tiny LLM Inference Engine.
 */

typedef struct
{
    int rows;
    int cols;
    float *data;
} Matrix;

/* Matrix memory management */
Matrix mat_create(int rows, int cols);
void mat_free(Matrix *m);

/* Matrix arithmetic */
Matrix mat_add(Matrix a, Matrix b);
Matrix mat_sub(Matrix a, Matrix b);
Matrix mat_scale(Matrix a, float scalar);
Matrix matmul(Matrix a, Matrix b);
Matrix mat_transpose(Matrix a);

/* Vector arithmetic */
float dot(const float *a, const float *b, int size);
void vector_add(const float *a, const float *b,
                float *result, int size);
void vector_sub(const float *a, const float *b,
                float *result, int size);
void vector_scale(const float *a, float scalar,
                  float *result, int size);

/* Statistical operations */
float mean(const float *data, int size);
float variance(const float *data, int size);

/* Mathematical transformations */
void exp_vector(const float *input, float *output, int size);
void softmax(const float *input, float *output, int size);
void normalize(const float *input, float *output, int size);

/* Transformer Layer Normalization */
void layer_norm(const float *input,
                float *output,
                const float *gamma,
                const float *beta,
                int size,
                float epsilon);

/* Display utilities */
void mat_print(Matrix m);
void vector_print(const float *data, int size);

#endif