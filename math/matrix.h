#ifndef MATRIX_H
#define MATRIX_H

// Stores the number of rows, columns, and the matrix data
typedef struct
{
    int rows;
    int cols;
    float *data;
} Matrix;

// Creates a matrix and allocates memory for its data
Matrix mat_create(int rows, int cols);

// Frees the memory used by a matrix
void mat_free(Matrix *m);

// Adds two matrices
Matrix mat_add(Matrix a, Matrix b);

// Subtracts two matrices
Matrix mat_sub(Matrix a, Matrix b);

// Multiplies every element of a matrix by a scalar
Matrix mat_scale(Matrix a, float scalar);

// Multiplies two matrices
Matrix matmul(Matrix a, Matrix b);

// Transposes a matrix
Matrix mat_transpose(Matrix a);

// Calculates the dot product of two vectors
float dot(const float *a, const float *b, int size);

// Adds two vectors
void vector_add(
    const float *a,
    const float *b,
    float *result,
    int size
);

// Subtracts two vectors
void vector_sub(
    const float *a,
    const float *b,
    float *result,
    int size
);

// Multiplies every element of a vector by a scalar
void vector_scale(
    const float *a,
    float scalar,
    float *result,
    int size
);

// Calculates the mean of a vector
float mean(const float *data, int size);

// Calculates the variance of a vector
float variance(const float *data, int size);

// Calculates e^x for every element of a vector
void exp_vector(
    const float *input,
    float *output,
    int size
);

// Applies min-max normalization to a vector
void normalize(
    const float *input,
    float *output,
    int size
);

// Prints a matrix
void mat_print(Matrix m);

// Prints a vector
void vector_print(
    const float *data,
    int size
);

#endif