#include "tensor.h"

int main(void)
{
    Tensor a = tensor_create(2, 3);

    if (a.data == NULL) {
        return 1;
    }

    for (size_t i = 0; i < a.rows * a.cols; i++) {
        a.data[i] = (float)(i + 1);
    }

    tensor_print(&a);
    tensor_free(&a);

    return 0;
}