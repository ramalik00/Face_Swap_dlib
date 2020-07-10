import cv2
from argparse import ArgumentParser
import dlib
from face_swap import swap_faces
import imutils

ap=ArgumentParser()
ap.add_argument("-i","--input1",type=str,required=True)
ap.add_argument("-m","--input2",type=str,default="")
ap.add_argument("-o","--output",type=str,default="")
args=vars(ap.parse_args())

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


image=cv2.imread(args["input1"])
cap=cv2.VideoCapture(args["input2"])
while True:
    (access,frame)=cap.read()
    frame=imutils.resize(frame,width=900)
    if not access:
        break

    result=swap_faces(image,frame,detector,predictor)

    cv2.imshow("frame",result)
    key=cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
    

    
    
