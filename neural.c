#include <stdio.h>
#include <math.h>

#define INPUTS 2
#define HIDDEN 2
#define SAMPLES 5

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double binary_cross_entropy(double y, double prediction) {
    const double epsilon = 1e-15;

    if (prediction < epsilon) {
        prediction = epsilon;
    }

    if (prediction > 1.0 - epsilon) {
        prediction = 1.0 - epsilon;
    }

    return -(y * log(prediction)
             + (1.0 - y) * log(1.0 - prediction));
}

int main(void) {
    /* Training data: study hours, attendance, result */
    double x[SAMPLES][INPUTS] = {
        {1.0, 0.3},
        {2.0, 0.4},
        {3.0, 0.5},
        {5.0, 0.8},
        {6.0, 0.9}
    };

    double y[SAMPLES] = {0, 0, 0, 1, 1};

    /* Weights and biases for hidden layer */
    double w[HIDDEN][INPUTS] = {
        {0.1, 0.2},
        {0.3, 0.1}
    };

    double b[HIDDEN] = {0.1, -0.2};

    /* Weights and bias for output layer */
    double v[HIDDEN] = {0.2, -0.1};
    double b_output = 0.1;

    double learning_rate = 0.01;
    int epochs = 10000;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double total_loss = 0.0;

        /* Store gradients for the whole dataset */
        double grad_w[HIDDEN][INPUTS] = {{0}};
        double grad_b[HIDDEN] = {0};

        double grad_v[HIDDEN] = {0};
        double grad_b_output = 0.0;

        for (int sample = 0; sample < SAMPLES; sample++) {
            double z[HIDDEN];
            double h[HIDDEN];

            /* Forward propagation through hidden layer */
            for (int i = 0; i < HIDDEN; i++) {
                z[i] = w[i][0] * x[sample][0]
                     + w[i][1] * x[sample][1]
                     + b[i];

                h[i] = relu(z[i]);
            }

            /* Forward propagation through output layer */
            double z_output = v[0] * h[0]
                            + v[1] * h[1]
                            + b_output;

            double prediction = sigmoid(z_output);

            /* Calculate loss */
            total_loss += binary_cross_entropy(
                y[sample],
                prediction
            );

            /* Backpropagation through output layer */
            double delta_output = prediction - y[sample];

            grad_v[0] += delta_output * h[0];
            grad_v[1] += delta_output * h[1];
            grad_b_output += delta_output;

            /* Backpropagation through hidden layer */
            for (int i = 0; i < HIDDEN; i++) {
                double delta_hidden =
                    delta_output * v[i] * relu_derivative(z[i]);

                grad_w[i][0] += delta_hidden * x[sample][0];
                grad_w[i][1] += delta_hidden * x[sample][1];
                grad_b[i] += delta_hidden;
            }
        }

        /* Average gradients over all training examples */
        for (int i = 0; i < HIDDEN; i++) {
            for (int j = 0; j < INPUTS; j++) {
                grad_w[i][j] /= SAMPLES;
            }

            grad_b[i] /= SAMPLES;
            grad_v[i] /= SAMPLES;
        }

        grad_b_output /= SAMPLES;
        total_loss /= SAMPLES;

        /* Update hidden layer parameters */
        for (int i = 0; i < HIDDEN; i++) {
            for (int j = 0; j < INPUTS; j++) {
                w[i][j] -= learning_rate * grad_w[i][j];
            }

            b[i] -= learning_rate * grad_b[i];
            v[i] -= learning_rate * grad_v[i];
        }

        /* Update output bias */
        b_output -= learning_rate * grad_b_output;

        if (epoch % 1000 == 0) {
            printf(
                "Epoch %d | Loss: %.4f\n",
                epoch,
                total_loss
            );
        }
    }

    /* Test the trained network */
    printf("\nPredictions after training:\n");

    for (int sample = 0; sample < SAMPLES; sample++) {
        double h[HIDDEN];

        for (int i = 0; i < HIDDEN; i++) {
            double z = w[i][0] * x[sample][0]
                     + w[i][1] * x[sample][1]
                     + b[i];

            h[i] = relu(z);
        }

        double z_output = v[0] * h[0]
                        + v[1] * h[1]
                        + b_output;

        double prediction = sigmoid(z_output);

        printf(
            "Hours: %.1f | Attendance: %.1f | "
            "Actual: %.0f | Prediction: %.4f | Class: %s\n",
            x[sample][0],
            x[sample][1],
            y[sample],
            prediction,
            prediction >= 0.5 ? "Pass" : "Fail"
        );
    }

    return 0;
}