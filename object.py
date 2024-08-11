import cv2
import numpy as np
import tensorflow as tf
import os

# Load the SSD MobileNet V2 model for object detection
modelFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Load the class labels
with open('coco_class_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Constants for display
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects

def display_text(im, text, x, y):
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)

def display_objects(im, objects, threshold=0.55):
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        if score > threshold:
            display_text(im, "{}: {:.2f}".format(labels[classId], score), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

def main():
    # Change this URL to the stream URL provided by your mobile device
    stream_url = 'http://172.20.10.2:8080/video'
    capture = cv2.VideoCapture(stream_url)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        objects = detect_objects(net, frame)
        display_objects(frame, objects, threshold=0.7)
        cv2.imshow("Object Detection and Path Finding", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
