#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"

// Creates a matrix and allocates memory for its elements
Matrix mat_create(int rows, int cols)
{
    Matrix m;

    m.rows = rows;
    m.cols = cols;

    m.data = malloc(rows * cols * sizeof(float));

    if (m.data == NULL)
    {
        printf("Memory allocation failed!\n");

        m.rows = 0;
        m.cols = 0;
    }

    return m;
}

// Releases the memory allocated for a matrix
void mat_free(Matrix *m)
{
    if (m->data != NULL)
    {
        free(m->data);
        m->data = NULL;
    }

    m->rows = 0;
    m->cols = 0;
}

// Adds two matrices of the same dimensions
Matrix mat_add(Matrix a, Matrix b)
{
    if (a.rows != b.rows || a.cols != b.cols)
    {
        printf("Error: matrix dimensions don't match!\n");

        return mat_create(0, 0);
    }

    Matrix result = mat_create(a.rows, a.cols);

    for (int i = 0; i < a.rows * a.cols; i++)
    {
        result.data[i] = a.data[i] + b.data[i];
    }

    return result;
}

// Subtracts the elements of matrix b from matrix a
Matrix mat_sub(Matrix a, Matrix b)
{
    if (a.rows != b.rows || a.cols != b.cols)
    {
        printf("Error: matrix dimensions don't match!\n");

        return mat_create(0, 0);
    }

    Matrix result = mat_create(a.rows, a.cols);

    for (int i = 0; i < a.rows * a.cols; i++)
    {
        result.data[i] = a.data[i] - b.data[i];
    }

    return result;
}

// Multiplies every element of a matrix by a scalar
Matrix mat_scale(Matrix a, float scalar)
{
    Matrix result = mat_create(a.rows, a.cols);

    for (int i = 0; i < a.rows * a.cols; i++)
    {
        result.data[i] = a.data[i] * scalar;
    }

    return result;
}

// Calculates the dot product of two vectors
float dot(const float *a, const float *b, int size)
{
    float result = 0.0f;

    for (int i = 0; i < size; i++)
    {
        result += a[i] * b[i];
    }

    return result;
}

// Adds two vectors element by element
void vector_add(
    const float *a,
    const float *b,
    float *result,
    int size
)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
}

// Subtracts two vectors element by element
void vector_sub(
    const float *a,
    const float *b,
    float *result,
    int size
)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] - b[i];
    }
}

// Multiplies every element of a vector by a scalar
void vector_scale(
    const float *a,
    float scalar,
    float *result,
    int size
)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] * scalar;
    }
}

// Multiplies two matrices using row-by-column multiplication
Matrix matmul(Matrix a, Matrix b)
{
    if (a.cols != b.rows)
    {
        printf(
            "Error: cannot multiply %dx%d by %dx%d\n",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        );

        return mat_create(0, 0);
    }

    Matrix result = mat_create(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < a.cols; k++)
            {
                // Access an element from the current row of matrix a
                float value_a =
                    a.data[i * a.cols + k];

                // Access an element from the current column of matrix b
                float value_b =
                    b.data[k * b.cols + j];

                sum += value_a * value_b;
            }

            // Store the calculated value in the result matrix
            result.data[i * result.cols + j] = sum;
        }
    }

    return result;
}

// Converts rows into columns and columns into rows
Matrix mat_transpose(Matrix a)
{
    Matrix result = mat_create(a.cols, a.rows);

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

// Calculates the average value of a vector
float mean(const float *data, int size)
{
    if (size <= 0)
        return 0.0f;

    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }

    return sum / size;
}

// Calculates how far the values are spread around their mean
float variance(const float *data, int size)
{
    if (size <= 0)
        return 0.0f;

    float avg = mean(data, size);

    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        // Calculate the difference between the value and the mean
        float difference = data[i] - avg;

        // Square the difference and add it to the total
        sum += difference * difference;
    }

    return sum / size;
}

// Calculates e^x for every element in a vector
void exp_vector(
    const float *input,
    float *output,
    int size
)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = expf(input[i]);
    }
}

// Normalizes values to a range between 0 and 1 using min-max normalization
void normalize(
    const float *input,
    float *output,
    int size
)
{
    if (size <= 0)
        return;

    float min = input[0];
    float max = input[0];

    // Find the minimum and maximum values
    for (int i = 1; i < size; i++)
    {
        if (input[i] < min)
            min = input[i];

        if (input[i] > max)
            max = input[i];
    }

    // Handle the case where all values are the same
    if (max == min)
    {
        for (int i = 0; i < size; i++)
        {
            output[i] = 0.0f;
        }

        return;
    }

    // Apply the min-max normalization formula
    for (int i = 0; i < size; i++)
    {
        output[i] =
            (input[i] - min) / (max - min);
    }
}

// Prints all elements of a matrix
void mat_print(Matrix m)
{
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            printf(
                "%8.3f ",
                m.data[i * m.cols + j]
            );
        }

        printf("\n");
    }
}

// Prints all elements of a vector
void vector_print(
    const float *data,
    int size
)
{
    printf("[ ");

    for (int i = 0; i < size; i++)
    {
        printf("%.3f ", data[i]);
    }

    printf("]\n");
}