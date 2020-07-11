import numpy as np
import cv2

def get_index(arr):
    index=None
    for num in arr[0]:
        index=num
        break
    return index
def triangulation(img,triangle,points):
    rect=cv2.boundingRect(triangle)
    (x,y,w,h)=rect
    cropped_triangle=img[y:y+h,x:x+w]
    cropped_tr_mask=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(cropped_tr_mask,points,255)
    cropped_triangle=cv2.bitwise_and(cropped_triangle,cropped_triangle,mask=cropped_tr_mask)
    return cropped_triangle
    

def swap_faces(img,img2,detector,predictor):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    mask=np.zeros_like(img_gray)
    img2_final_face=np.zeros_like(img2)
    faces=detector(img_gray)
    
    for face in faces:
        landmarks_points=[]
        landmarks=predictor(img_gray,face)
        for i in range(68):
            x=landmarks.part(i).x
            y=landmarks.part(i).y
            landmarks_points.append((x,y))
        
    points=np.array(landmarks_points,np.int32)
    convexhull=cv2.convexHull(points)
    cv2.fillConvexPoly(mask,convexhull,255)
    face_image_1=cv2.bitwise_and(img,img,mask=mask)


    
    #Delauney Triangulisation
    indexes_triangles=[]
    rect=cv2.boundingRect(convexhull)
    subdiv=cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles=subdiv.getTriangleList()
    triangles=np.array(triangles,dtype=np.int32)



    #stroring triangle coordinates
    for t in triangles:
        pt1=(t[0],t[1])
        pt2=(t[2],t[3])
        pt3=(t[4],t[5])
        index_pt1=np.where((points==pt1).all(axis=1))
        index_pt1=get_index(index_pt1)
        index_pt2=np.where((points==pt2).all(axis=1))
        index_pt2=get_index(index_pt2)
        index_pt3=np.where((points==pt3).all(axis=1))
        index_pt3=get_index(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            indexes_triangles.append([index_pt1,index_pt2,index_pt3])

            

    faces2=detector(img2_gray)
    landmarks_points2=[]
    for face in faces2:
        landmarks=predictor(img2_gray,face)
        for i in range(68):
            x=landmarks.part(i).x
            y=landmarks.part(i).y
            landmarks_points2.append((x,y))

            
    for triangle in indexes_triangles:
        #triangulation of first face
        tr1_pt1=landmarks_points[triangle[0]]
        tr1_pt2=landmarks_points[triangle[1]]
        tr1_pt3=landmarks_points[triangle[2]]
        triangle1=np.array([tr1_pt1,tr1_pt2,tr1_pt3],np.int32)
        rect1=cv2.boundingRect(triangle1)
        (x,y,w,h)=rect1
        cropped_triangle1=img[y:y+h,x:x+w]
        cropped_tr1_mask=np.zeros((h,w),np.uint8)
        points1=np.array([[tr1_pt1[0]-x,tr1_pt1[1]-y],[tr1_pt2[0]-x,tr1_pt2[1]-y],[tr1_pt3[0]-x,tr1_pt3[1]-y]])
        cv2.fillConvexPoly(cropped_tr1_mask,points1,255)
        cropped_triangle1=cv2.bitwise_and(cropped_triangle1,cropped_triangle1,mask=cropped_tr1_mask)
        


        #triangulation of second face
        tr2_pt1=landmarks_points2[triangle[0]]
        tr2_pt2=landmarks_points2[triangle[1]]
        tr2_pt3=landmarks_points2[triangle[2]]
        triangle2=np.array([tr2_pt1,tr2_pt2,tr2_pt3],np.int32)
        rect2=cv2.boundingRect(triangle2)
        (x,y,w,h)=rect2
        cropped_triangle2=img2[y:y+h,x:x+w]
        cropped_tr2_mask=np.zeros((h,w),np.uint8)
        points2=np.array([[tr2_pt1[0]-x,tr2_pt1[1]-y],[tr2_pt2[0]-x,tr2_pt2[1]-y],[tr2_pt3[0]-x,tr2_pt3[1]-y]])
        cv2.fillConvexPoly(cropped_tr2_mask,points2,255)
        cropped_triangle2=cv2.bitwise_and(cropped_triangle2,cropped_triangle2,mask=cropped_tr2_mask)




        #Wrapping triangles on face 2
        points1=np.float32(points1)
        points2=np.float32(points2)
        M=cv2.getAffineTransform(points1,points2)
        warped_triangle=cv2.warpAffine(cropped_triangle1,M,(w,h))
        img2_final_face_rect=img2_final_face[y:y+h,x:x+w]
        img2_final_face_rect_gray=cv2.cvtColor(img2_final_face_rect,cv2.COLOR_BGR2GRAY)
        _,mask_triangles_designed=cv2.threshold(img2_final_face_rect_gray,1,255,cv2.THRESH_BINARY_INV)
        warped_triangle=cv2.bitwise_and(warped_triangle,warped_triangle,mask=mask_triangles_designed)



        #Reconstruct face2 making sure black spaces are not overlapped
        triangle_area=img2_final_face[y:y+h,x:x+w]
        triangle_area=cv2.add(triangle_area,warped_triangle)
        img2_final_face[y:y+h,x:x+w]=triangle_area
        

        #swapping face    
    img2_final_face=cv2.medianBlur(img2_final_face,5)
    img2_final_face_gray=cv2.cvtColor(img2_final_face,cv2.COLOR_BGR2GRAY)
    _,background=cv2.threshold(img2_final_face_gray,1,255,cv2.THRESH_BINARY)
    r=cv2.boundingRect(background)
    center=((r[0]+r[2]//2,r[1]+r[3]//2))
    result=cv2.seamlessClone(img2_final_face,img2,background,center,cv2.NORMAL_CLONE)
    

    return result
