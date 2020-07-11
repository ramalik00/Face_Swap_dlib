# Face_Swap_dlib

### Requirements
    1.Opencv 4.3.0
    2.Dlib 19.20.0
    3.Imutils
    4.Python 3.4 and above
    Clone the repository and also download the shape predictor file from this link:http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
### Functionality:
    The script uses dlib's get_frontal_face_detector() to detect face and shape predictor model to get the landmarks on the face and then 
    perfrom face swapping using these landmark points.
    
### Content:
    • face_swap.py file contains all the functions and code to do face swapping
    • In main.py you have to pass two arguments 
       1. --input1 <image1>  Path of input image 1
       2. --input2 <image2> Path of input image 2
    Image 1 corresponds to that person who's you would swap with that of person in Image 2
    • In main_video.py you have to pass only 1 required argument 
      1.  --input1 <image1>  Path of input image 1
    By default your system's camera will access the input 2    
    
