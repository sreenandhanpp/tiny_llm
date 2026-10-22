
#include <stdio.h>
#include <math.h>
#include "matrix.h"

#define EPSILON 0.001f

static int check_value(const char *name, float actual, float expected)
{
    if (fabsf(actual - expected) > EPSILON)
    {
        printf("FAIL: %s | expected %.3f, got %.3f\n",
               name, expected, actual);
        return 0;
    }

    printf("PASS: %s\n", name);
    return 1;
}

int main(void)
{
    int passed = 0;
    int total = 0;

    /* Test matrix multiplication: (2x2) * (2x2). */
    Matrix a = mat_create(2, 2);
    Matrix b = mat_create(2, 2);

    if (a.data == NULL || b.data == NULL)
    {
        printf("Matrix allocation failed.\n");
        mat_free(&a);
        mat_free(&b);
        return 1;
    }

    a.data[0] = 1; a.data[1] = 2;
    a.data[2] = 3; a.data[3] = 4;

    b.data[0] = 5; b.data[1] = 6;
    b.data[2] = 7; b.data[3] = 8;

    Matrix product = matmul(a, b);

    total++;
    if (product.data != NULL)
    {
        int ok = 1;
        ok &= check_value("matmul[0,0]", product.data[0], 19);
        ok &= check_value("matmul[0,1]", product.data[1], 22);
        ok &= check_value("matmul[1,0]", product.data[2], 43);
        ok &= check_value("matmul[1,1]", product.data[3], 50);
        passed += ok;
    }
    else
    {
        printf("FAIL: matrix multiplication allocation\n");
    }

    /* Test matrix transpose. */
    Matrix transposed = mat_transpose(a);

    total++;
    if (transposed.data != NULL)
    {
        int ok = 1;
        ok &= check_value("transpose[0,0]", transposed.data[0], 1);
        ok &= check_value("transpose[0,1]", transposed.data[1], 3);
        ok &= check_value("transpose[1,0]", transposed.data[2], 2);
        ok &= check_value("transpose[1,1]", transposed.data[3], 4);
        passed += ok;
    }
    else
    {
        printf("FAIL: transpose allocation\n");
    }

    /* Test vector dot product. */
    float x[] = {1, 2, 3};
    float y[] = {4, 5, 6};

    total++;
    passed += check_value("dot product", dot(x, y, 3), 32);

    /* Test numerically stable softmax. */
    float logits[] = {1000, 1001, 1002};
    float probabilities[3];

    softmax(logits, probabilities, 3);

    total++;
    int softmax_ok = 1;
    softmax_ok &= check_value("softmax[0]", probabilities[0], 0.09003f);
    softmax_ok &= check_value("softmax[1]", probabilities[1], 0.24473f);
    softmax_ok &= check_value("softmax[2]", probabilities[2], 0.66524f);
    passed += softmax_ok;

    total++;
    passed += check_value(
        "softmax probability sum",
        probabilities[0] + probabilities[1] + probabilities[2],
        1.0f
    );

    mat_free(&a);
    mat_free(&b);
    mat_free(&product);
    mat_free(&transposed);

    printf("\nResults: %d/%d tests passed.\n", passed, total);

    return passed == total ? 0 : 1;
}
