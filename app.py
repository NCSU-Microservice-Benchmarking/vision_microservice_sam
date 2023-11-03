from ultralytics import SAM
import torch
import time
import cv2
import numpy as np
from flask import request
from flask import Flask
from flask import Response
import os
import requests

app = Flask(__name__)

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

# initial the model
# model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model.eval()

@app.route('/greet')
def greet():
    return("Hello")


@app.route('/count_gpu')
def countGPU():
        return str(torch.cuda.device_count())

@app.route('/detections',methods=['POST'])
def detect():
    # get image from the request form
    image = request.form.get('image')
    # convert image string to a numpy array
    # image_buffer = bytes(image, 'utf-8')
    # image = np.frombuffer(image_buffer, np.uint8)
    # print(image.shape)
    # # image = request.files.get('image')
    # # # convert image string to a numpy array
    # # image = np.fromstring(image, np.uint8)
    # # convert the image into a opencv image
    # image = cv2.imdecode(image, -1)
    # send the image to the model and get the result
    # Display model information (optional)
    model.info()

    # Run inference
    results = model(image)
    
    # get the bounding boxes of the detected objects
    # boxes = results.xyxy[0].cpu().numpy()
    # model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
    # draw the bounding boxes on the image
    # for box in boxes:
        # x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        # image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        # put the confidence value and category name on the image
        # image = cv2.putText(image,str(box[4]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    # convert image to a binary string
    image = cv2.imencode('.png', image)[1].tostring()
    # send the image to the client as an png image encoded as a binary string
    return Response(image, mimetype='image/png')

# a route to test the server where allows to specify the ip and port of the test server
# example: http://10.161.0.29:8000/test?test_server_ip=localhost&test_server_port=8001&task=detection&self_port=8000&api=detections
@app.route('/test')
def test():
    test_server_ip = request.args.get('test_server_ip')
    test_server_port = request.args.get('test_server_port')
    task = request.args.get('task')
    self_port = request.args.get('self_port')
    api = request.args.get('api')
    print("test server ip: " + test_server_ip)
    print("test server port: " + test_server_port)
    print("task: " + task)
    print("self port: " + self_port)
    print("api: " + api)
    # send a post request to the test server
    # with args "task" to note the task it is performing, "port" of the it is running on, and "api" to specify the api to test
    url = "http://" + test_server_ip + ":" + test_server_port + "/tests?task=" + task + "&port=" + self_port + "&api=" + api
    print("request url: " + url)
    # send the request
    response = requests.post(url)
    # check the response
    if response.status_code == 200:
        return "Test successful"
    else:
        return "Test failed"