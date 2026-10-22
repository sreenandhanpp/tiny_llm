
/*
 * matrix.c
 *
 * Implements matrix arithmetic, vector operations, statistical
 * calculations, normalization, and utilities for the Tiny LLM
 * Inference Engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "matrix.h"

/*
 * Checks whether a matrix has valid dimensions and allocated
 * storage before performing operations on its elements.
 */
static int matrix_valid(Matrix m)
{
    return m.data != NULL && m.rows > 0 && m.cols > 0;
}

/*
 * Creates a matrix and allocates contiguous memory for its
 * elements. All elements are initialized to zero.
 *
 * An invalid matrix is returned if dimensions are invalid,
 * allocation fails, or the allocation size would overflow.
 */
Matrix mat_create(int rows, int cols)
{
    Matrix m = {0, 0, NULL};

    if (rows <= 0 || cols <= 0)
    {
        return m;
    }

    if ((size_t)rows > SIZE_MAX / (size_t)cols)
    {
        return m;
    }

    size_t count = (size_t)rows * (size_t)cols;

    if (count > SIZE_MAX / sizeof(float))
    {
        return m;
    }

    m.data = calloc(count, sizeof(float));

    if (m.data == NULL)
    {
        return m;
    }

    m.rows = rows;
    m.cols = cols;

    return m;
}

/*
 * Releases the memory owned by a matrix and resets its fields.
 * Passing a NULL pointer is safe.
 */
void mat_free(Matrix *m)
{
    if (m == NULL)
    {
        return;
    }

    free(m->data);

    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

/*
 * Adds two matrices element by element.
 * Both matrices must have identical dimensions.
 *
 * The caller owns the returned matrix and must free it.
 */
Matrix mat_add(Matrix a, Matrix b)
{
    if (!matrix_valid(a) || !matrix_valid(b) ||
        a.rows != b.rows || a.cols != b.cols)
    {
        return (Matrix){0, 0, NULL};
    }

    Matrix result = mat_create(a.rows, a.cols);

    if (!matrix_valid(result))
    {
        return result;
    }

    int count = a.rows * a.cols;

    for (int i = 0; i < count; i++)
    {
        result.data[i] = a.data[i] + b.data[i];
    }

    return result;
}

/*
 * Subtracts matrix b from matrix a element by element.
 * Both matrices must have identical dimensions.
 */
Matrix mat_sub(Matrix a, Matrix b)
{
    if (!matrix_valid(a) || !matrix_valid(b) ||
        a.rows != b.rows || a.cols != b.cols)
    {
        return (Matrix){0, 0, NULL};
    }

    Matrix result = mat_create(a.rows, a.cols);

    if (!matrix_valid(result))
    {
        return result;
    }

    int count = a.rows * a.cols;

    for (int i = 0; i < count; i++)
    {
        result.data[i] = a.data[i] - b.data[i];
    }

    return result;
}

/*
 * Multiplies every element of a matrix by a scalar.
 */
Matrix mat_scale(Matrix a, float scalar)
{
    if (!matrix_valid(a))
    {
        return (Matrix){0, 0, NULL};
    }

    Matrix result = mat_create(a.rows, a.cols);

    if (!matrix_valid(result))
    {
        return result;
    }

    int count = a.rows * a.cols;

    for (int i = 0; i < count; i++)
    {
        result.data[i] = a.data[i] * scalar;
    }

    return result;
}

/*
 * Calculates the dot product of two vectors.
 *
 * The result is the sum of the products of corresponding
 * elements. Both vectors must contain at least size elements.
 */
float dot(const float *a, const float *b, int size)
{
    if (a == NULL || b == NULL || size <= 0)
    {
        return 0.0f;
    }

    float result = 0.0f;

    for (int i = 0; i < size; i++)
    {
        result += a[i] * b[i];
    }

    return result;
}

/*
 * Adds corresponding vector elements and stores the results
 * in the output vector.
 */
void vector_add(
    const float *a,
    const float *b,
    float *result,
    int size)
{
    if (a == NULL || b == NULL || result == NULL || size <= 0)
    {
        return;
    }

    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
}

/*
 * Subtracts corresponding vector elements and stores the
 * results in the output vector.
 */
void vector_sub(
    const float *a,
    const float *b,
    float *result,
    int size)
{
    if (a == NULL || b == NULL || result == NULL || size <= 0)
    {
        return;
    }

    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] - b[i];
    }
}

/*
 * Multiplies every vector element by a scalar.
 */
void vector_scale(
    const float *a,
    float scalar,
    float *result,
    int size)
{
    if (a == NULL || result == NULL || size <= 0)
    {
        return;
    }

    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] * scalar;
    }
}

/*
 * Multiplies two matrices using row-by-column multiplication.
 *
 * If a has dimensions m x n and b has dimensions n x p,
 * the result has dimensions m x p.
 *
 * Each output element is the dot product of one row of a
 * and one column of b.
 */
Matrix matmul(Matrix a, Matrix b)
{
    if (!matrix_valid(a) || !matrix_valid(b) ||
        a.cols != b.rows)
    {
        return (Matrix){0, 0, NULL};
    }

    Matrix result = mat_create(a.rows, b.cols);

    if (!matrix_valid(result))
    {
        return result;
    }

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < a.cols; k++)
            {
                float value_a = a.data[i * a.cols + k];
                float value_b = b.data[k * b.cols + j];

                sum += value_a * value_b;
            }

            result.data[i * result.cols + j] = sum;
        }
    }

    return result;
}

/*
 * Transposes a matrix by exchanging its rows and columns.
 *
 * An input matrix with dimensions m x n produces an
 * output matrix with dimensions n x m.
 */
Matrix mat_transpose(Matrix a)
{
    if (!matrix_valid(a))
    {
        return (Matrix){0, 0, NULL};
    }

    Matrix result = mat_create(a.cols, a.rows);

    if (!matrix_valid(result))
    {
        return result;
    }

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[j * result.cols + i] =
                a.data[i * a.cols + j];
        }
    }

    return result;
}

/*
 * Calculates the arithmetic mean of a vector.
 */
float mean(const float *data, int size)
{
    if (data == NULL || size <= 0)
    {
        return 0.0f;
    }

    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }

    return sum / (float)size;
}

/*
 * Calculates population variance.
 *
 * Population variance is the average squared difference
 * between each element and the mean.
 */
float variance(const float *data, int size)
{
    if (data == NULL || size <= 0)
    {
        return 0.0f;
    }

    float avg = mean(data, size);
    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        float difference = data[i] - avg;
        sum += difference * difference;
    }

    return sum / (float)size;
}

/*
 * Calculates the exponential of every vector element.
 *
 * This function is useful for general mathematical operations.
 * For softmax, use the numerically stable softmax() function.
 */
void exp_vector(
    const float *input,
    float *output,
    int size)
{
    if (input == NULL || output == NULL || size <= 0)
    {
        return;
    }

    for (int i = 0; i < size; i++)
    {
        output[i] = expf(input[i]);
    }
}

/*
 * Converts a vector of logits into probabilities.
 *
 * Subtracting the maximum input before exponentiation reduces
 * the risk of numerical overflow.
 *
 * The output probabilities sum approximately to one.
 * Input and output may point to the same vector.
 *
 * Inputs must contain finite floating-point values.
 */
void softmax(
    const float *input,
    float *output,
    int size)
{
    if (input == NULL || output == NULL || size <= 0)
    {
        return;
    }

    float max_value = input[0];

    for (int i = 1; i < size; i++)
    {
        if (input[i] > max_value)
        {
            max_value = input[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        output[i] = expf(input[i] - max_value);
        sum += output[i];
    }

    if (sum <= 0.0f || !isfinite(sum))
    {
        return;
    }

    for (int i = 0; i < size; i++)
    {
        output[i] /= sum;
    }
}

/*
 * Applies min-max normalization to map values into [0, 1].
 *
 * This is useful for data preprocessing, but it is not
 * the LayerNorm operation used by Transformer blocks.
 *
 * If all values are equal, the output is filled with zeros.
 */
void normalize(
    const float *input,
    float *output,
    int size)
{
    if (input == NULL || output == NULL || size <= 0)
    {
        return;
    }

    float min_value = input[0];
    float max_value = input[0];

    for (int i = 1; i < size; i++)
    {
        if (input[i] < min_value)
        {
            min_value = input[i];
        }

        if (input[i] > max_value)
        {
            max_value = input[i];
        }
    }

    if (max_value == min_value)
    {
        for (int i = 0; i < size; i++)
        {
            output[i] = 0.0f;
        }

        return;
    }

    float range = max_value - min_value;

    for (int i = 0; i < size; i++)
    {
        output[i] = (input[i] - min_value) / range;
    }
}

/*
 * Applies Layer Normalization to a single vector.
 *
 * gamma contains learnable scale parameters.
 * beta contains learnable bias parameters.
 * epsilon prevents division by zero.
 *
 * The input vector is normalized using its population
 * mean and variance:
 *
 * output[i] = gamma[i] * (input[i] - mean)
 *             / sqrt(variance + epsilon) + beta[i]
 *
 * The parameters must match the trained model.
 */
void layer_norm(
    const float *input,
    float *output,
    const float *gamma,
    const float *beta,
    int size,
    float epsilon)
{
    if (input == NULL || output == NULL ||
        gamma == NULL || beta == NULL ||
        size <= 0 || epsilon <= 0.0f)
    {
        return;
    }

    float avg = mean(input, size);
    float var = variance(input, size);
    float denominator = sqrtf(var + epsilon);

    for (int i = 0; i < size; i++)
    {
        float normalized = (input[i] - avg) / denominator;

        output[i] = gamma[i] * normalized + beta[i];
    }
}

/*
 * Prints a matrix in row-and-column form.
 */
void mat_print(Matrix m)
{
    if (!matrix_valid(m))
    {
        printf("Invalid or empty matrix.\n");
        return;
    }

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            printf("%8.3f ", m.data[i * m.cols + j]);
        }

        printf("\n");
    }
}

/*
 * Prints all elements of a vector.
 */
void vector_print(const float *data, int size)
{
    if (data == NULL || size <= 0)
    {
        printf("[]\n");
        return;
    }

    printf("[ ");

    for (int i = 0; i < size; i++)
    {
        printf("%.3f ", data[i]);
    }

    printf("]\n");
}
