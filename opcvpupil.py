#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')#On error change to absolute path
font = cv2.FONT_HERSHEY_SIMPLEX


# In[3]:


facePos = np.array([[0,0,0,0],[0,0,0,0]])
eyePos = np.array([0,0,0,0])
staticPos = np.array([0,0,0,0])
staticEyePos = np.array([0,0,0,0])
isStatic = False
settled = False
def sharpen(image):
    kernel = np.array([[0,-1,0],[-1,4.5,-1],[0,-1,0]],np.float32)#Laplace算子实现滤波器
    #Actually array:[[0,-1,0],[-1,5,-1],[0,-1,0]] works as sharpening,this kernel in use destroys the image,but makes pupil more outstanding.
    dst = cv2.filter2D(image,-1,kernel=kernel)
    return dst


# In[4]:


while(cap.isOpened()):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    shp = sharpen(frame)
    shp = cv2.GaussianBlur(shp,(7,7),0)
    gshp = cv2.cvtColor(shp, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gshp,55,100)
    
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        facePos = faces
        eyes = eye_cascade.detectMultiScale(roi_gray)
        try:
            if(len(eyes)>=1 or eyes!=None):
                eyePos = list(eyes)
        except:
            pass
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(roi_color,(int(ex+ew/2),int(ey+eh/2)),2,(0,255,0),-1)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            pass
    
    i = 0
    try:
        if(isStatic==False):
            settled = False
        for (eex,eey,eew,eeh) in eyePos:
            cv2.putText(frame,f'posX={eex},posY={eey},w={eew},h={eeh}',(10,300-i*30),font,0.8,(255,255,255),2)
            i=i+1
    except:
        pass
    i=1
    if(isStatic):
        if(settled == False):
            settled = True
            staticEyePos = eyePos
            staticPos = facePos
        cv2.putText(frame,'Static',(10,330),font,0.8,(255,255,255),2)
        for (eex,eey,eew,eeh) in staticEyePos:
            cv2.putText(frame,f'Static Eye Pos(Rlt to Face),posX={eex},posY={eey},w={eew},h={eeh}',(10,330+i*30),font,0.8,(255,255,255),2)
            i=i+1
        cv2.putText(frame,f'Static Face Position,posX={staticPos[0][0]},posY={staticPos[0][1]}',(10,330+i*30),font,0.8,(255,255,255),2)
    ret,thresh = cv2.threshold(canny,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ii=0
    
    for index,contr in enumerate(contours):
        M=cv2.moments(contr)
        if(M['m00'] != 0):
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(contr)
        else:
            continue
        for po in eyePos:
            frame = cv2.rectangle(frame,(po[0]+facePos[0][0],po[1]+facePos[0][1]),(po[0]+po[2]+facePos[0][0],po[1]+po[3]+facePos[0][1]),(0,0,255),2)
            if(x>po[0]+facePos[0][0] and y>po[1]+facePos[0][1] and x+w<po[0]+po[2]+facePos[0][0] and y+h<po[1]+po[3]+facePos[0][1]):
                frame = cv2.drawContours(frame,contours,index,(0,0,255),3)
    #cv2.drawContours(frame,contours,-1,(0,0,255),3)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        isStatic = bool(1-isStatic)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[5]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




