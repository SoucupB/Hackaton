#include <stdio.h>
#include "NeuralNetwork.h"
#include <time.h>
#include <math.h>
#include "Functions.h"
#include <stdlib.h>

int main() {
    // XOR problem!
    func_UseSrand();
    int32_t maxIterations = 9408;
    NeuralNetwork network;
    int32_t inputs[] = {2, 5, 1};
    float input[5][5] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float output[5][5] = {{0}, {1}, {1}, {0}};
    float arcInput[9] = {0, 0, 0, 1, 1, 0, 1, 1};
    float arcOutput[5] = {0, 1, 1, 0};
    int32_t functionsIndex[] = {RELU, SIGMOID};
    network = nn_InitMetaParameters(inputs, 3, 0.1, functionsIndex);
    printf("Started!\n");
    long mil = func_Time();
    for(int32_t i = 0; i < maxIterations; i++) {
        printf("%f\n", nn_TrainBigBuffer(network, arcInput, 2, arcOutput, 1, 4, OPT_SGD, 5));
    }
    printf("Done in: %ld miliseconds\n", func_Time() - mil);
    printf("Responses:\n");
    for(int32_t j = 0; j < 4; j++) {
        float *response = nn_FeedForward(network, input[j], 2);
        printf("Response for inputs [%.1f, %.1f] is %f\n", input[j][0], input[j][1], response[0]);
    }
    nn_Destroy(network);
    printf("DONE");
}