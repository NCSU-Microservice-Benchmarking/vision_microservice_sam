from ultralytics import yolov8
from PIL import Image
import cv2
import numpy as np

# Load a model
model = yolov8('yolov8n-seg.pt')

# Display model information (optional)
model.info()

# # Run inference and save as an image
results = model('bus.png')
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
# save results as a PNG image with opencv
cv2.imwrite('sam_results.png', im_array)
    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    # im.save('sam_results.jpg')  # save image


def test_detect():
    # get image from the request form
    image = cv2.IMREAD('bus.png')
    image = image.read()
    # convert image string to a numpy array
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    # image = np.frombuffer(image_buffer, np.uint8)
    # print(image.shape)
    # image = request.files.get('image')
    # # # convert image string to a numpy array
    # # image = np.fromstring(image, np.uint8)
    # # convert the image into a opencv image
    # image = cv2.imdecode(image, -1)
    # send the image to the model and get the result
    # Display model information (optional)
    # Run inference 
    results = model(image)
    # for r in results:
        # image = r.plot()  # plot a BGR numpy array of predictions
    
    # get the bounding boxes of the detected objects
    boxes = results.xyxy[0].cpu().numpy()
    # model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
    # draw the bounding boxes on the image
    for box in boxes:
        x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        # put the confidence value and category name on the image
        image = cv2.putText(image,str(box[4]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    # save the image to local as "result.png"
    cv2.imwrite("result.png", image)
    # convert image to a binary string
    image = cv2.imencode('.png', image)[1].tostring()
    # send the image to the client as an png image encoded as a binary string
    return Response(image, mimetype='image/png')