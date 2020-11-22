#include "Neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Neuron ne_Init(int32_t ID, struct hashmap *hash,
               float (*activationFunction)(float), float (*derivativeActivationFunction)(float), float lr) {
    Neuron neuron = malloc(sizeof(struct Neuron_t));
    neuron->childs = malloc(sizeof(struct Neuron_t) * MAX_CONNECTIONS);
    neuron->parents = malloc(sizeof(struct Neuron_t) * MAX_CONNECTIONS);
    for(int32_t i = 0; i < MAX_CONNECTIONS; i++) {
        neuron->childs[i] = NULL;
        neuron->parents[i] = NULL;
    }
    neuron->value = 0;
    neuron->error = 0;
    neuron->ID = ID;
    neuron->hash = hash;
    neuron->unChangedValue = 0;
    neuron->childsCount = 0;
    neuron->parentsCount = 0;
    neuron->shouldApplyActivation = 1;
    neuron->activationFunction = activationFunction;
    neuron->derivativeActivationFunction = derivativeActivationFunction;
    neuron->lr = lr;
    neuron->adaptiveLr = 0;
    neuron->epsilon = 1e-5;
    neuron->beta = 0.9;
    neuron->momentum = 0;
    neuron->beta1 = 0.9;
    neuron->beta2 = 0.999;
    return neuron;
}

void ne_Destroy(Neuron neuron) {
    free(neuron);
}

void ne_Tie(Neuron parent, Neuron child, float value) {
    parent->childs[parent->childsCount++] = child;
    child->parents[child->parentsCount++] = parent;
    saveWeight(parent->hash, parent->ID, child->ID, value);
    saveWeight(parent->hash, child->ID, parent->ID, value);
}

void ne_FeedForward(Neuron neuron) {
    if(neuron->shouldApplyActivation) {
        ne_Activate(neuron);
    }
    for(int32_t i = 0; i < neuron->childsCount; i++) {
        Neuron child = neuron->childs[i];
        child->value += neuron->value * getWeight(neuron->hash, neuron->ID, child->ID);
        child->unChangedValue = child->value;
    }
}

void ne_Activate(Neuron neuron) {
    assert(neuron->activationFunction != NULL);
    neuron->value = neuron->activationFunction(neuron->value);
}

void ne_PropagateErrorToParents(Neuron self) {
    for(int32_t i = 0; i < self->parentsCount; i++) {
        Neuron parent = self->parents[i];
        parent->error += self->error * getWeight(self->hash, self->ID, parent->ID);
    }
}

void ne_OptimizeSGD(Neuron self) {
    for(int32_t i = 0; i < self->parentsCount; i++) {
        Neuron parent = self->parents[i];
        float gradient = getWeight(parent->hash, parent->ID, self->ID) +
                         self->error * self->lr * self->derivativeActivationFunction(self->unChangedValue) * parent->value;
        saveWeight(parent->hash, self->ID, parent->ID, gradient);
        saveWeight(parent->hash, parent->ID, self->ID, gradient);
    }
}