#include <stdio.h>
#include "NeuralNetwork.h"
#include <time.h>
#include <math.h>
#include "Functions.h"
#include <stdlib.h>
int main() {
    // XOR problem with neuroevolution!
    int32_t maxIterations = 60070;
    int32_t inputs[] = {2, 4, 1};
    float input[5][5] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float output[5][5] = {{0}, {1}, {1}, {0}};
    int32_t maxNets = 30;
    int32_t functionsIndex[] = {RELU, SIGMOID};
    NeuralNetwork net = nn_InitMetaParameters(inputs, 3, 0.2, functionsIndex);
    printf("Started!\n");
    for(int32_t i = 0; i < 100; i++) {
        printf("%f\n", nn_Optimize(net, input[i], 2, output[i], 1, OPT_SGD));
    }
    printf("DONE");
}