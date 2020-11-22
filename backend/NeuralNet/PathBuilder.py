import os
import sys
sitePackae = -1
pythonPath = ""
print("Info: Adding library to the path directory...")
for pathVar in sys.path:
    sitePackae = pathVar.find("site-packages", 0)
    if sitePackae != -1:
        pythonPath = pathVar
if pythonPath == "":
    print("Error: site-packages not found!")
    exit()
try:
    fd = open(r"" + pythonPath + r'\NeuralNetwork.pth', 'w+')
    fd.write(os.path.dirname(os.path.abspath(__file__)))
    fd.close()
    print("Succesfull: Added library to the package library!")
except IOError:
    print("Could not create path file")
