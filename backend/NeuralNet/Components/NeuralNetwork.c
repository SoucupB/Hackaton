#include "NeuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

NeuralNetwork nn_InitMetaParameters(int32_t *structureBuffer, int32_t size, float lr, int32_t *configuration) {
    NeuralNetwork neuralNetwork = malloc(sizeof(struct NeuralNetwork_t));
    int32_t *hiddensSizes = malloc(sizeof(int32_t) * size);
    neuralNetwork->hiddensSizes = hiddensSizes;
    struct hashmap *hash = initBufferHash();
    struct Function_t *functions = func_GetActivationFunctions();
    struct Function_t *dFunctions = func_GetDActivationFunctions();
    neuralNetwork->hash = hash;
    neuralNetwork->lr = lr;
    neuralNetwork->numberOfHiddens = 0;
    neuralNetwork->functions = functions;
    neuralNetwork->dFunctions = dFunctions;
    int32_t ids = 0;
    Neuron *layer = malloc(sizeof(Neuron) * structureBuffer[0]);
    Neuron *allNeurons = malloc(sizeof(Neuron) * (func_ArraySum(structureBuffer, size) + size - 1));
    int32_t neuronsIndex = 0;
    neuralNetwork->biases = malloc(sizeof(struct Neuron_t) * size);
    for(int32_t j = 0; j < structureBuffer[0]; j++) {
        layer[j] = ne_Init(ids++, hash, NULL, NULL, neuralNetwork->lr);
        layer[j]->shouldApplyActivation = 0;
        allNeurons[neuronsIndex++] = layer[j];
    }
    neuralNetwork->hiddensSizes[0] = structureBuffer[0];
    neuralNetwork->inputs = layer;
    for(int32_t i = 1; i < size; i++) {
        layer = malloc(sizeof(Neuron) * structureBuffer[i]);
        for(int32_t j = 0; j < structureBuffer[i]; j++) {
            layer[j] = ne_Init(ids++, hash, functions[configuration[i - 1]].func, dFunctions[configuration[i - 1]].func, neuralNetwork->lr);
            allNeurons[neuronsIndex++] = layer[j];
        }
        neuralNetwork->hiddens[neuralNetwork->numberOfHiddens++] = layer;
        neuralNetwork->hiddensSizes[i] = structureBuffer[i];
    }
    for(int32_t i = 0; i < neuralNetwork->hiddensSizes[0]; i++) {
        for(int32_t j = 0; j < neuralNetwork->hiddensSizes[1]; j++) {
            ne_Tie(neuralNetwork->inputs[i], neuralNetwork->hiddens[0][j], func_Uniform(MIN_INTERVAR, MAX_INTERVAR));
        }
    }
    for(int32_t i = 0; i < neuralNetwork->numberOfHiddens - 1; i++) {
        for(int32_t j = 0; j < neuralNetwork->hiddensSizes[i + 1]; j++) {
            for(int32_t k = 0; k < neuralNetwork->hiddensSizes[i + 2]; k++) {
                ne_Tie(neuralNetwork->hiddens[i][j], neuralNetwork->hiddens[i + 1][k], func_Uniform(MIN_INTERVAR, MAX_INTERVAR));
            }
        }
    }
    for(int32_t i = 0; i < neuralNetwork->numberOfHiddens; i++) {
        neuralNetwork->biases[i] = ne_Init(ids++, hash, functions[configuration[i]].func, dFunctions[configuration[i]].func, neuralNetwork->lr);
        neuralNetwork->biases[i]->value = 1.0;
        neuralNetwork->biases[i]->shouldApplyActivation = 0;
        allNeurons[neuronsIndex++] = neuralNetwork->biases[i];
    }
    for(int32_t i = 0; i < neuralNetwork->numberOfHiddens; i++) {
        for(int32_t j = 0; j < neuralNetwork->hiddensSizes[i + 1]; j++) {
            ne_Tie(neuralNetwork->biases[i], neuralNetwork->hiddens[i][j], func_Uniform(MIN_INTERVAR, MAX_INTERVAR));
        }
    }
    neuralNetwork->maxNeurons = ids;
    neuralNetwork->allNeurons = allNeurons;
    return neuralNetwork;
}

void nn_ShowWeights(NeuralNetwork net) {
    assert(net->numberOfHiddens > 0);
    for(int32_t i = 0; i < net->hiddensSizes[0]; i++) {
        for(int32_t j = 0; j < net->hiddensSizes[1]; j++) {
            printf("%f ", getWeight(net->hash, net->inputs[i]->ID, net->hiddens[0][j]->ID));
        }
        printf("\n");
    }
    printf("\n");
    for(int32_t i = 0; i < net->numberOfHiddens - 1; i++) {
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            for(int32_t k = 0; k < net->hiddensSizes[i + 2]; k++) {
                printf("%f ", getWeight(net->hash, net->hiddens[i][j]->ID, net->hiddens[i + 1][k]->ID));
            }
            printf("\n");
        }
    }
}

void nn_ClearNeurons(NeuralNetwork net) {
    for(int32_t i = 0; i < net->numberOfHiddens; i++) {
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            net->hiddens[i][j]->value = 0;
            net->hiddens[i][j]->error = 0;
            net->hiddens[i][j]->unChangedValue = 0;
        }
    }
}

void nn_Mutate(NeuralNetwork self, float chance, float by) {
    int32_t numberOfMutations = (int32_t)(chance * (float)self->maxNeurons) + 1.0;
    assert(self->maxNeurons != 0);
    for(int32_t i = 0; i < numberOfMutations; i++) {
        int32_t fNode = (int32_t)func_RandomNumber(0, (float)(self->maxNeurons) - 0.001);
        Neuron *nodes = NULL;
        int32_t size = 0;
        if(self->allNeurons[fNode]->parentsCount > 0) {
            nodes = self->allNeurons[fNode]->parents;
            size = self->allNeurons[fNode]->parentsCount;
        }
        else
        if(self->allNeurons[fNode]->childsCount > 0) {
            nodes = self->allNeurons[fNode]->childs;
            size = self->allNeurons[fNode]->childsCount;
        }
        int32_t sNode = (int32_t)func_RandomNumber(0, (float)(size) - 0.001);
        float weight = getWeight(self->hash, fNode, nodes[sNode]->ID);
        float delta = func_RandomNumber(-by, by);
        saveWeight(self->hash, fNode, nodes[sNode]->ID, weight + delta);
    }
}

float *nn_FeedForward(NeuralNetwork net, float *structureBuffer, int32_t size) {
    float *result = malloc(sizeof(float) * size);
    nn_ClearNeurons(net);
    for(int32_t i = 0; i < size; i++) {
        net->inputs[i]->value = structureBuffer[i];
        net->inputs[i]->unChangedValue = structureBuffer[i];
        net->inputs[i]->error = 0;
        ne_FeedForward(net->inputs[i]);
    }
    for(int32_t i = 0; i < net->numberOfHiddens; i++) {
        ne_FeedForward(net->biases[i]);
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            ne_FeedForward(net->hiddens[i][j]);
        }
    }
    for(int32_t i = 0; i < net->hiddensSizes[net->numberOfHiddens]; i++) {
        result[i] = net->hiddens[net->numberOfHiddens - 1][i]->value;
    }
    return result;
}

float nn_Optimize(NeuralNetwork net, float *input, int32_t inputSize, float *output, int32_t outputSize, int8_t type) {
    float *inputResponse = NULL;
    inputResponse = nn_FeedForward(net, input, inputSize);
    float totalError = 0;
    for(int32_t i = 0; i < outputSize; i++) {
        float valueError = func_SquaredError(output[i], inputResponse[i]);
        net->hiddens[net->numberOfHiddens - 1][i]->error = valueError;
        totalError += fabs(valueError);
    }
    for(int32_t i = net->numberOfHiddens - 1; i >= 0; i--) {
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            ne_PropagateErrorToParents(net->hiddens[i][j]);
        }
    }
    if(type == OPT_SGD) {
        for(int32_t i = net->numberOfHiddens - 1; i >= 0; i--) {
            for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
                ne_OptimizeSGD(net->hiddens[i][j]);
            }
        }
    }
    free(inputResponse);
    return totalError;
}

float nn_TrainBigBuffer(NeuralNetwork net, float *input, int32_t inputSize, float *output, int32_t outputSize,
                        int32_t size, int8_t type, int32_t batches) {
    float totalCost = 0;
    for(int32_t j = 0; j < batches; j++) {
        for(int32_t i = 0; i < size; i++) {
            totalCost += nn_Optimize(net, &input[inputSize * i], inputSize, &output[outputSize * i], outputSize, type);
        }
    }
    return totalCost;
}

void nn_WriteFile(NeuralNetwork net) {
    FILE *fd = fopen(FILE_NAME, "w+");
    fprintf(fd, "%d\n", (int32_t)hashmap_count(net->hash));
    for(int32_t i = 0; i < net->maxNeurons; i++) {
        for(int32_t j = 0; j < net->maxNeurons; j++) {
            float weight = getWeight(net->hash, i, j);
            if(weight) {
                fprintf(fd, "%d %d %f\n", i, j, weight);
            }
        }
    }
    fclose(fd);
}

void nn_LoadFile(NeuralNetwork network) {
    FILE *fd = fopen(FILE_NAME, "r+");
    int32_t totalWeights, a, b;
    float c;
    fscanf(fd, "%d", &totalWeights);
    for(int32_t i = 0; i < totalWeights; i++) {
        fscanf(fd, "%d %d %f", &a, &b, &c);
        saveWeight(network->hash, a, b, c);
    }
    fclose(fd);
}

float elementFromBuffer(float *buffer, int32_t index) {
    return buffer[index];
}

void nn_Destroy(NeuralNetwork net) {
    for(int32_t j = 0; j < net->maxNeurons; j++) {
        ne_Destroy(net->allNeurons[j]);
    }
    free(net->hiddensSizes);
    free(net->biases);
    hashmap_free(net->hash);
    free(net->allNeurons);
    free(net->functions);
    free(net->dFunctions);
    free(net);
}