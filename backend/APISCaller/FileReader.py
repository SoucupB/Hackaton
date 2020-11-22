import numpy as np
import cv2
def GetImageData():
    fd = open("MnistData/train-images.idx3-ubyte", 'rb')
    data = fd.read()
    ar = int.from_bytes(data[4:8], byteorder='big')
    rows = int.from_bytes(data[8:12], byteorder='big')
    cols = int.from_bytes(data[12:16], byteorder='big')
    offset = 16
    dataFiles = []
    for imageOffset in range(ar):
        img = []
        for pixels in range(rows * cols):
            img.append(data[offset + pixels + imageOffset * rows * cols] / 255.0)
        dataFiles.append(img)
    fd.close()
    return dataFiles

def GetLabels():
    fd = open("MnistData/train-labels.idx1-ubyte", 'rb')
    data = fd.read()
    ar = int.from_bytes(data[4:8], byteorder='big')
    offset = 8
    dataFiles = []
    for labelOffset in range(ar):
        label = [0] * 10
        label[data[labelOffset + offset]] = 1
        dataFiles.append(label)
    fd.close()
    return dataFiles

def ShowImage(image, width, height):
    img = np.zeros((height, width, 1),np.uint8)
    for i in range(height):
        for j in range(width):
            img[i, j] = (image[i * height + j] * 255.0)
    cv2.imwrite("Image.png",img)
    return 0

def createImageFromArray(buffer, height, width, nH, nW):
    img = np.zeros((height, width, 1),np.uint8)
    newBuffer = []
    for i in range(height):
        for j in range(width):
            img[i, j] = (buffer[i * height + j])
    img = cv2.resize(img, (nH, nW), interpolation = cv2.INTER_AREA)
    for i in range(nH):
        for j in range(nW):
            newBuffer.append(img[i, j] / 255.0)
    return newBuffer