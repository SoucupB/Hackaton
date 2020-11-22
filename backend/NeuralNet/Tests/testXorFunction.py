import NeuralNetwork as nn
import time
import math
#initialize neural Network.
network = nn.NeuralNetwork([2, 5, 1], 0.1, [nn.RELU, nn.RELU])
millis = int(round(time.time() * 1000))
total_error = 0.05
print("Info: Test (Xor trainer) executed in", (int(round(time.time() * 1000)) - millis) / 1000.0, "seconds!")
inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
outputs = [[0], [1], [1], [0]]
for index in range(300):
    network.sgdLong(inputs, outputs, 20, nn.OPT_SGD)
tests_response = 0
for test in range(len(inputs)):
    tests_response += math.fabs(outputs[test][0] - network.feed_forward(inputs[test])[0])
print("Info: Maximum addmited error is", total_error)
if tests_response < total_error:
    print("Success: Neural Network error precision on Xor Problem is %.2f" % tests_response)
else:
    print("Error: Neural Network error is to big!")