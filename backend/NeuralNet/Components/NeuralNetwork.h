#pragma once
#include <stdint.h>
#include "Neuron.h"
#include "hashmap.h"
#include <assert.h>
#include "Functions.h"
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define IDENTITY 3
#define SOFTPLUS 4
#define ARCTAN 5
#define GAUSSIAN 6
#define MIN_INTERVAR -0.01
#define MAX_INTERVAR 0.01
#define FILE_NAME "networks.rt"

#define OPT_SGD 8
#define OPT_SGDM 9
#define OPT_SGDNM 10
#define OPT_ADAGRAD 11
#define OPT_ADAM 12

struct NeuralNetwork_t;
typedef struct NeuralNetwork_t* NeuralNetwork;

struct NeuralNetwork_t
{
    Neuron *hiddens[1<<8];
    Neuron *biases;
    Neuron *inputs;
    Neuron *allNeurons;
    int32_t *hiddensSizes;
    int32_t numberOfHiddens;
    float lr;
    struct hashmap *hash;
    struct Function_t *functions;
    struct Function_t *dFunctions;
    int32_t maxNeurons;
};

NeuralNetwork nn_InitMetaParameters(int32_t *structureBuffer, int32_t size, float lr, int32_t *configuration);
float *nn_FeedForward(NeuralNetwork net, float *structureBuffer, int32_t size);
void nn_ShowWeights(NeuralNetwork net);
float nn_Optimize(NeuralNetwork net, float *input, int32_t inputSize, float *output, int32_t outputSize, int8_t type);
void nn_ClearNeurons(NeuralNetwork net);
void nn_Destroy(NeuralNetwork net);
void nn_WriteFile(NeuralNetwork net);
void nn_LoadFile(NeuralNetwork network);
void nn_CrossOver(NeuralNetwork first, NeuralNetwork second);
void nn_Mutate(NeuralNetwork self, float chance, float by);
float nn_TrainBigBuffer(NeuralNetwork net, float *input, int32_t inputSize, float *output, int32_t outputSize,
                        int32_t size, int8_t type, int32_t batches);

float elementFromBuffer(float *buffer, int32_t index);