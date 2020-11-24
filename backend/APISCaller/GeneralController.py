from flask import render_template, jsonify, Flask, request
from flask_cors import CORS, cross_origin
import NeuralNetwork as nn
from io import StringIO
import sys
import FileReader as fr

app = Flask(__name__)
app.config['SECRET_KEY'] = 'djfdhsgshuighyygtyftyftJHHGHGfdjsfidshgjushgusgh'
app.config['CORS_HEADERS'] = 'Content-Type'


cors = CORS(app, resources={r"/code": {"origins": "http://localhost:4040"}})

def extract_code(string):
    code = ""
    for line in string:
        code += line
        code += "\n"
    return code

@app.route('/execute.json', methods = ['POST'])
@cross_origin(origin='localhost',headers=['Content- Type'])
def show_steps():
    json_inp = request.get_json()
    code = extract_code(json_inp["code"])
    codeOut = StringIO()
    sys.stdout = codeOut
    exec(code)
    sys.stdout = sys.__stdout__
    s = codeOut.getvalue()
    s = s.split('\n')
    s.pop()
    return jsonify({"response": s})

@app.route('/test.json', methods = ['GET'])
@cross_origin(origin='localhost',headers=['Content- Type'])
def getMetaData():
    s = 0
    return jsonify({"response": s})

def getNumberIndex(response):
    max = 0
    ind = 0
    for index in range(len(response)):
        if max < response[index]:
            max = response[index]
            ind = index
    return ind

@app.route('/mnistPost.json', methods = ['POST'])
@cross_origin(origin='localhost',headers=['Content- Type'])
def runMnistData():
    json_inp = request.get_json()
    newImg = fr.createImageFromArray(json_inp['data'], json_inp['height'], json_inp['width'], 28, 28)
    fr.ShowImage(newImg, 28, 28)
    net = nn.NeuralNetwork([28 * 28, 51, 10], 0.2, [nn.RELU, nn.SIGMOID])
    data = net.feed_forward(newImg)
    net.load_weights()
    response = getNumberIndex(data)
    return jsonify({"response": response, "raw_data": data})
if __name__ == '__main__':
    app.run(host= '192.168.100.34', threaded=True, port=5000)