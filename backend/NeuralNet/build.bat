@echo off
:start
echo Compiling...
python --version 3>NUL
if errorlevel 1 goto errorNoPython
gcc -fPIC -shared Components/NeuralNetwork.c Components/hashmap.c Components/Functions.c Components/Neuron.c -Wall -o  Components/NeuralNetwork.so -O9
if exist Components/NeuralNetwork.so (echo Succesfull: Library built successfully) else (echo gcc is probably missing, install it and add it to the environment variables.)
if exist Components/NeuralNetwork.so (python PathBuilder.py)
if exist Components/NeuralNetwork.so (echo Info: Building tests...)
if exist Components/NeuralNetwork.so (python Tests/testXorFunction.py)
Exit /B 5
:errorNoPython
echo Error^: Python not installed
