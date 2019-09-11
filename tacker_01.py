# imports
from imageai.Detection import ObjectDetection
import os
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import math

# params
model_type = "fast" # "accurate", "normal", "fast"
speed = "fast" # "normal"(default), "fast", "faster" , "fastest" and "flash"
confidence_person = 30
confidence_face = 0
font = cv2.FONT_HERSHEY_SIMPLEX
# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
args = vars(ap.parse_args())

# loading models
execution_path = os.getcwd()
detector = ObjectDetection()
if(model_type == "accurate"):
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
elif model_type == "fast":
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolo-tiny.h5"))
else:
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed=speed)
custom = detector.CustomObjects(person=True)
# init age and gender parameters 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

# load our serialized model from disk
print("[INFO] loading models...")
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"]) #for face-detection
age_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_age.prototxt", 
                        "age_gender_models/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_gender.prototxt", 
                        "age_gender_models/gender_net.caffemodel")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# if the video argument is None, then the code will read from webcam (work in progress)
if args.get("video", None) is None:
    cap = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    cap = cv2.VideoCapture(args["video"])
writer = None
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
     
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
ct1 = CentroidTracker()
(H, W) = (None, None)
font = cv2.FONT_HERSHEY_SIMPLEX
# loop over the frames from the video stream
person = 0
person1 = 0

while True:
	# read the next frame from the video stream and resize it
    frame = cap.read()
    frame = frame if args.get("video", None) is None else frame[1]
    # if the frame can not be grabbed, then we have reached the end of the video
    if frame is None:
        break
    # if the frame dimensions are None, grab them
    if W is None or H is None or writer is None:
        (H, W) = frame.shape[:2]
        if args.get("output", None) is not None:
            out_file = os.path.splitext(args["output"])[0].replace("\\", "") + ".avi"
        elif args.get("video", None) is not None:
            out_file = "out" + os.path.splitext(args["video"])[0].replace("\\", "") + ".avi"
        else:
            out_file = "output.avi"
        # print(out_file)
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        writer = cv2.VideoWriter(out_file, fourcc, 1.3*fps, (W, H), True)
    
    # begin detection frame by frame
    rects = []
    face_rects = []
    ages = []
    genders = []
    detected_image_array, detections = detector.detectCustomObjectsFromImage(input_type="array", output_type="array", custom_objects=custom, input_image=frame, minimum_percentage_probability=confidence_person)
    # cv2.imshow("Frame", frame)
    for obj in detections:
        (startX, startY, endX, endY) = obj["box_points"]
        rects.append(obj["box_points"])
        w = endX - startX
        h = endY - startY
        s = 1
        if min(w, h) < 300:
            s = int(300/min(w, h))
        print("s = ", s)
        resized = cv2.resize(frame, (s*W, s*H), interpolation = cv2.INTER_CUBIC)
        person_img = resized[s*startY:s*endY, s*startX:s*endX].copy()
        blob = cv2.dnn.blobFromImage(person_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        net.setInput(blob)
        detections = net.forward()
        rects1 = []
        confidence = []
        for i in range(0, detections.shape[2]):
            # box1 = detections[0, 0, i, 3:7] * np.array([endY-startY, endX-startX, endY-startY, endX-startX])
            box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sX, sY, eX, eY) = box1.astype("int")
            if(sX<0 or sY<0 or eX>w or eY>h or sY + eY > h or eX < sX + 5 or eY < sY + 5 or eY-sY>2.5*(eX-sX) or eY-sY<0.8*(eX-sX)): continue
            # conf = 0
            # i1 = 0
            # while i1<10:
            #     xx = 2*(sX + i1*(eX-sX)/10)
            #     i1 = i1 + 1
            #     j1 = 0
            #     while j1<10:
            #         yy = 2*(sY + j1*(eY-sY)/10)
            #         conf += detections[0, 0, i, 2]*math.exp(-((xx - w)**2/(0.9*w) + (yy - h/8)**2/(0.5*h)))
            #         j1 = j1 + 1
            
            # if(sY + eY < h/4): conf = 10*conf
            rects1.append([sX, sY, eX, eY])
            confidence.append(detections[0, 0, i, 2])
            # cv2.rectangle(frame, (sX, sY), (eX, eY),(155, 55, 200), 1)
        
        # cv2.rectangle(frame, (startX, startY), (endX, endY),(5, 55, 55), 4)
        if len(confidence) == 0:
            genders.append('face not recognized')
            ages.append(' ')  
            continue
        max_i = confidence.index(max(confidence))
        (sX, sY, eX, eY) = rects1[max_i]
        # drawInner = drawInner + 1
        face_rects.append([sX+startX, sY+startY, eX+startX, eY+startY])
        cv2.rectangle(frame, (sX+startX, sY+startY), (eX+startX, eY+startY),(255, 255, 255), 1)
        w1 = s*(eX - sX)
        h1 = s*(eY - sY)
        x = s*sX
        y = s*sY
        s1 = 1
        if min(w1, h1) < 300:
            s1 = int(300/min(w1, h1))
        print("s1 = ", s1)
        resized1 = cv2.resize(person_img, (s1*s*w, s1*s*h), interpolation = cv2.INTER_CUBIC)
        face_img = resized1[s1*y:s1*(y+h1), s1*x:s1*(x+w1)].copy()
        try:
            blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        except:
            print("Something went wrong -- (sy, ey, sx, ex) = (", startY, "', '", endY, "', '", startX, ", ", endX, ")")
            genders.append('front face not found')
            ages.append(' ')                
            continue
            # Predict gender
        gender_net.setInput(blob2)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        # Predict age
        age_net.setInput(blob2)
        age_preds = age_net.forward()
        # print(age_preds)
        age = age_list[age_preds[0].argmax()]
        genders.append(gender)
        ages.append(age)
        
	# update our centroid tracker using the computed set of bounding
	# box rectangles
    objects = ct.update(rects)
	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        if(objectID>person):
            # cv2.imwrite('img/'+str(person)+'.jpg',detected_image_array[person])
            person = person+1
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)
    # update our centroid tracker for face using the computed set of bounding
	# box rectangles
    # objects1 = ct1.update(face_rects)
	# # loop over the tracked objects
    # for (objectID, centroid) in objects1.items():
	# 	# draw both the ID of the object and the centroid of the
	# 	# object on the output frame
    #     text = "ID {}".format(objectID)
    #     if(objectID>person1):
    #         # cv2.imwrite('img/'+str(person)+'.jpg',detected_image_array[person])
    #         person1 = person1+1
    #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)

    i = 0
    while i < len(rects):
        rect = rects[i]
        gender = genders[i]
        age = ages[i]
        (startX, startY, endX, endY) = rect
        cv2.rectangle(frame, (startX, startY), (endX, endY),(155, 255, 0), 2)
        overlay_text = "%s, %s" % (gender, age)
        if age != ' ':
            cv2.putText(frame, overlay_text, (startX,startY), font, 0.6, (255,0,0), 2, cv2.LINE_4)
        else:
            cv2.putText(frame, overlay_text, (startX,startY), font, 0.5, (0,0,255), 1, cv2.LINE_4)
        i = i + 1

    if writer is not None:
        writer.write(frame)
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
