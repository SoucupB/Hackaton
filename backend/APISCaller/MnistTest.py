import NeuralNetwork as nn
import FileReader as fr
images = fr.GetImageData()
labels = fr.GetLabels()
imageHeight = 28
imageWidth = 28
epochs = 20
net = nn.NeuralNetwork([imageHeight * imageWidth, 51, 10], 0.2, [nn.RELU, nn.SIGMOID])
for i in range(20):
    print(i, net.sgdLong(images[:(50 * (i + 1))], labels[:(50 * (i + 1))], 10, nn.OPT_SGD))

net.save_weights()
print(net.feed_forward(images[4]))
print(labels[4])
