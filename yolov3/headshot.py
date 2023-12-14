import cv2
import argparse
import numpy as np
import imutils
import time
import pyautogui
from mss import mss
from PIL import Image
import time

bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

sct = mss()


'''webcam = cv2.VideoCapture(0)

writer = None
(W, H) = (None, None)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()'''
    
classes = None
with open("yolo.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# try to determine the total number of frames in the video file
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h,center_point):

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    

# run inference through the network
while True:
    sct_img = sct.grab(bounding_box)
    sct_image=np.array(sct_img)
    frame = cv2.cvtColor(sct_image, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, (416, 416))
    (Width,Height)=(416,416)
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    center_point = []
    temp = 0.5
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5   :
            
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                center_point.append([center_x, center_y,class_id])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = 0
# go through the detections remaining
# after nms and draw bounding box
    for i in indices:
        i = i
        
        time.sleep(0.3)
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        pyautogui.moveTo(int(center_point[i][0]*1920/416), int((center_point[i][1]-h/2.9)*1080/416))
        draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),center_point[detections])
        detections+=1
    cv2.imshow("object detection", frame)
    
    #pyautogui.click(int(center_point[0][0]*1920/416), int(center_point[0][2]*1080/416))
# wait until any key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


   

    
