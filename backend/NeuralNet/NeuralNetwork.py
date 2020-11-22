import ctypes
import math
import time
import os
SIGMOID = 0
TANH = 1
RELU = 2
IDENTITY = 3
SOFTPLUS = 4
ARCTAN = 5
GAUSSIAN = 6

OPT_SGD = 8
OPT_SGDM = 9
OPT_SGDNM = 10
OPT_ADAGRAD = 11
OPT_ADAM = 12

class NeuralNetwork():
    def __init__(self, input_array, lr, configuration):
        path = os.path.dirname(os.path.abspath(__file__))
        self.fun = ctypes.CDLL(r"" + path + r"\Components\NeuralNetwork.so")
        self.input_array = input_array
        self.lr = lr
        self.configuration = configuration
        ints_array = ctypes.c_int * len(input_array)
        configuration_array = ctypes.c_int * len(configuration)
        raw_array = ints_array(*self.input_array)
        configuration_array = configuration_array(*self.configuration)
        c_lr = ctypes.c_float(self.lr)
        self.fun.func_UseSrand()
        self.neuralNet = self.fun.nn_InitMetaParameters(raw_array, len(self.input_array), c_lr, configuration_array)
        self.fun.elementFromBuffer.restype = ctypes.c_float
        self.fun.nn_Optimize.restype = ctypes.c_float
        self.fun.func_Uniform.restype = ctypes.c_float
        self.fun.nn_TrainBigBuffer.restype = ctypes.c_float
    def show_weights(self):
        self.fun.nn_ShowWeights(self.neuralNet)

    def buffer_to_list(self, buffer, size):
        return_buffer = []
        for index in range(size):
            return_buffer.append(self.fun.elementFromBuffer(buffer, ctypes.c_int(index)))
        return return_buffer

    def feed_forward(self, inputs):
        c_inputs = ctypes.c_float * len(inputs)
        input_array = c_inputs(*inputs)
        response = self.fun.nn_FeedForward(self.neuralNet, input_array, len(input_array))
        arr = ctypes.c_float * 1
        list_of_results = self.buffer_to_list(response, self.input_array[len(self.input_array) - 1])
        self.fun.func_FreePointer(response)
        return list_of_results

    def sgd(self, input, output, optimization_method=OPT_SGD):
        c_inputs = ctypes.c_float * len(input)
        input_array = c_inputs(*input)

        c_output = ctypes.c_float * len(output)
        output_array = c_output(*output)
        return self.fun.nn_Optimize(self.neuralNet, input_array, len(input), output_array, len(output), ctypes.c_char(optimization_method))

    def sgdLong(self, input, output, batches, optimization_method=OPT_SGD):
        bufferCount = len(input)
        inputSize = len(input[0])
        outputSize = len(output[0])
        flat_input = [item for sublist in input for item in sublist]
        c_inputs = ctypes.c_float * len(flat_input)
        input_array = c_inputs(*flat_input)
        flat_output = [item for sublist in output for item in sublist]
        c_output = ctypes.c_float * len(flat_output)
        output_array = c_output(*flat_output)
        return self.fun.nn_TrainBigBuffer(self.neuralNet, input_array, inputSize, output_array, outputSize,
                                          ctypes.c_int(bufferCount), ctypes.c_char(optimization_method), ctypes.c_int(batches))

    def destroy_nn(self):
        self.fun.nn_Destroy(self.neuralNet)

    def save_weights(self):
        self.fun.nn_WriteFile(self.neuralNet)

    def load_weights(self):
        self.fun.nn_LoadFile(self.neuralNet)

    def get_random(self, a, b):
        return self.fun.func_Uniform(ctypes.c_float(a), ctypes.c_float(b))