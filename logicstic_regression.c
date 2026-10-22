#include <stdio.h>

int main()
{
    // Training data
    float x[] = {1, 2, 3, 4, 5};      // Study hours
    float y[] = {35, 45, 55, 65, 75}; // Actual marks

    int n = 5;

    // Model parameters
    float weight = 0;
    float bias = 0;

    // Learning rate
    float learning_rate = 0.01;

    // Number of training iterations
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float dw = 0;
        float db = 0;

        // Go through every training example
        for (int i = 0; i < n; i++)
        {
            // Prediction
            float prediction = weight * x[i] + bias;

            // Error
            float error = prediction - y[i];

            // Calculate gradients
            dw += error * x[i];
            db += error;
        }

        // Average the gradients
        dw = dw / n;
        db = db / n;

        // Update parameters
        weight = weight - learning_rate * dw;
        bias = bias - learning_rate * db;
    }

    printf("Learned weight = %.2f\n", weight);
    printf("Learned bias   = %.2f\n", bias);

    // Make a prediction
    float hours = 6;

    float prediction = weight * hours + bias;

    printf("Prediction for %.0f hours = %.2f marks\n",
           hours, prediction);

    return 0;
}