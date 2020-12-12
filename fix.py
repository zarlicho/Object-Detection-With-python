import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)
wht = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))
modelConfiguraction = 'yolov3.cfg'
modelWeights = 'yolov3-tiny.weights'
net = cv.dnn.readNetFromDarknet(modelConfiguraction, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def finObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    #print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        print(x, y)
        start_point = (0,0)
        end_point = (x,y)
        color = (0,255,255)
        lebar = 2
        image = cv.line(img,start_point,end_point,color,lebar)

while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img,1/255,(wht, wht),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    finObjects(outputs, img)
    cv.imshow('frame', img)
    cv.waitKey(1)
