import cv2
from argparse import ArgumentParser
import dlib
from face_swap import swap_faces

ap=ArgumentParser()
ap.add_argument("-i","--input1",type=str,required=True)
ap.add_argument("-m","--input2",type=str,required=True)
ap.add_argument("-o","--output",type=str,default="")
args=vars(ap.parse_args())

image1=cv2.imread(args["input1"])
image2=cv2.imread(args["input2"])

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

result=swap_faces(image1,image2,detector,predictor)



if args["output"]!="":
    cv2.imwrite(args["output"],result)

cv2.imshow("frame",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
    

