#include <stdio.h>
#include "NeuralNetwork.h"
#include <time.h>
#include <math.h>
#include "Functions.h"
#include "QAgent.h"

// void testQTraining(QAgent agent) {
//     float inputArray[][16] = {
//                         {
//                             0, 0, 0,
//                             0, 1, 0,
//                             0, 0, 0
//                         },
//                         {
//                             0, 0, -1,
//                             0, 1, 1,
//                             0, 0, 0
//                         },
//                         {
//                             0, 0, -1,
//                             -1, 1, 1,
//                             0, 1, 0
//                         },
//                         {
//                             0, -1, -1,
//                             -1, 1, 1,
//                             0, 1, 1
//                         }
//     };
//     int32_t finalReward = 1.0;
//     int32_t actionsTaken[] = {2, 3, 1, 0};
//     qa_TrainTemporalDifference(agent, inputArray, actionsTaken, finalReward, 4);
//     printf("Done training!\n");
// }

int main() {
    // TicTacToe board qa testing!.
    NeuralNetwork network;
    float netLr = 0.1;
    float trainLr = 0.2;
    int32_t maxIterations = 10000;
    int32_t statesSize = 9, nrOfActions = 9;
    int32_t inputs[] = {statesSize + nrOfActions, 18, 1};
    int32_t layersNumber = 3;
    int32_t possibleActionsNumber = 9;
    float trainingLearningRate = trainLr, discountFactor = 0.1;
    int32_t activationFunctions[] = {RELU, SIGMOID};
    network = nn_InitMetaParameters(inputs, layersNumber, netLr, activationFunctions);
    QAgent agent = qa_Init(network, trainingLearningRate, discountFactor, possibleActionsNumber);
    float inputArray[] = {
                            0, 0, 1,
                            0, 0, 0,
                            0, 0, 0
                        };
    int32_t prohibitedAction[] = {2};
    long mil = func_Time();
    for(int32_t i = 0; i < maxIterations; i++) {
        qa_GetChoosenActionIndex(agent, inputArray, prohibitedAction, 1);
    }
    printf("Choosed %d actions based on the neural nets guess on this tictactoe board!\n", maxIterations);
    printf("Done in: %ld miliseconds\n", func_Time() - mil);
    //testQTraining(agent);
    qa_Destroy(agent);
    nn_Destroy(network);
    printf("Memory freed!");
}