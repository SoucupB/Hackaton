var radius;
var c;
var imge = []
var width;
var height;
var url = "http://localhost:5000/mnistPost.json";

function htmlToElement(html) {
    var template = document.createElement('template');
    html = html.trim();
    template.innerHTML = html;
    return template.content.firstChild;
}

function start_debugg() {
    loadPixels();
    sentData = []
    for(var i = 0; i < pixels.length; i += 4) {
        sentData.push(255 - pixels[i]);
    }
    console.log(sentData.length);
    httpPost(url, 'json', {"height": height, "width": width, "data": sentData}, function(response) {
        console.log(response);
        var stre = "<p>The network estimates that this number is " + response["response"].toString() + "</p>";
        var element;
        var el = document.getElementById("mesg");
        if(el != null) {
            var element = el;
            element.innerHTML = stre;
        }
        else {
            element = htmlToElement(stre);
            element.id = "mesg"
        }
        c_element = document.getElementById("resp");
        c_element.appendChild(element);
    });
    return false;
}

function setup() {
    width = 390;
    height = 390;
    cnv = createCanvas(height, width);
    cnv.parent('ExampleBox')
    createP();
    c = color(0, 0, 0);
    background(255);
    colorMode(RGB, 255)
}

function draw() {
  radius = 20;
}

function mouseClicked() {
  if (mouseX > 400) {
    c = get(mouseX, mouseY);
  }
}

function executeCode() {
    var newUrl = "http://localhost:5000/execute.json";
    var value = document.getElementById("txtarea").value.split("\n")
    console.log(value)
    httpPost(newUrl, 'json', {"code": value}, function(response) {
        var resp = "";
        for(var i = 0; i < response["response"].length; i++) {
            resp += response["response"][i] + "\n";
        }
        document.getElementById("output").innerHTML = resp
        console.log(response)
    });
}

function mouseDragged() {
    stroke(c)
    if (mouseX < width) {
        strokeWeight(radius);
        imge.push(mouseY * height + mouseX)
        imge.push(pmouseY * height + pmouseX)
        line(mouseX, mouseY, pmouseX, pmouseY);
    }
}

function changeBG() {
  background(255);
  imge = []
  return false;
}
