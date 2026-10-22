#include <stdio.h>
#include <math.h>

/* Sigmoid function converts any number into a value between 0 and 1 */
float sigmoid(float z)
{
    return 1.0 / (1.0 + exp(-z));
}

int main()
{
    /* Training data */
    float x[] = {1, 2, 3, 4, 5, 6};
    float y[] = {0, 0, 0, 1, 1, 1};

    int n = 6;

    /* Model parameters */
    float weight = 0;
    float bias = 0;

    /* Training settings */
    float learning_rate = 0.1;
    int epochs = 10000;

    /* Training */
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float dw = 0;
        float db = 0;
        float loss = 0;

        /* Go through every training example */
        for (int i = 0; i < n; i++)
        {
            /* Linear calculation */
            float z = weight * x[i] + bias;

            /* Convert z into probability */
            float prediction = sigmoid(z);

            /* Calculate error */
            float error = prediction - y[i];

            /* Calculate gradients */
            dw += error * x[i];
            db += error;

            /* Calculate binary cross-entropy loss */
            loss += -(y[i] * log(prediction)
                    + (1 - y[i]) * log(1 - prediction));
        }

        /* Average gradients */
        dw = dw / n;
        db = db / n;
        loss = loss / n;

        /* Update parameters */
        weight = weight - learning_rate * dw;
        bias = bias - learning_rate * db;

        /* Print progress */
        if (epoch % 1000 == 0)
        {
            printf("Epoch: %d  Loss: %.4f  Weight: %.4f  Bias: %.4f\n",
                   epoch, loss, weight, bias);
        }
    }

    printf("\nTraining completed!\n");

    printf("Weight = %.4f\n", weight);
    printf("Bias   = %.4f\n", bias);

    /* Make a prediction */
    float hours;

    printf("\nEnter study hours: ");
    scanf("%f", &hours);

    /* Linear calculation */
    float z = weight * hours + bias;

    /* Convert to probability */
    float probability = sigmoid(z);

    printf("Probability of passing = %.2f%%\n",
           probability * 100);

    /* Classification */
    if (probability >= 0.5)
        printf("Prediction: PASS\n");
    else
        printf("Prediction: FAIL\n");

    return 0;
}