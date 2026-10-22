#include <stdio.h>

int main()
{
    // 1. Create an array
    float weights[5] = {0.5, 1.2, -0.7, 3.4, 2.1};

    // 2. Save the array to model.bin
    FILE *file = fopen("model.bin", "wb");

    if (file == NULL)
    {
        printf("Error opening file!\n");
        return 1;
    }

    fwrite(weights, sizeof(float), 5, file);

    fclose(file);

    printf("Model saved!\n");


    // 3. Create another array for loading
    float loaded_weights[5];

    // 4. Load the model
    file = fopen("model.bin", "rb");

    if (file == NULL)
    {
        printf("Error opening file!\n");
        return 1;
    }

    fread(loaded_weights, sizeof(float), 5, file);

    fclose(file);


    // 5. Print the loaded values
    printf("\nLoaded weights:\n");

    for (int i = 0; i < 5; i++)
    {
        printf("weight[%d] = %.2f\n", i, loaded_weights[i]);
    }

    return 0;
}