import numpy as np
import os
from PIL import Image
import cv2


type="none"
dir="D:/FYP/whole_face/"+str(type)
eye_cascade = cv2.CascadeClassifier('D:/FYP/code/haarcascade_eye.xml')


for file_name in os.listdir(dir):
    print(file_name)
    file_path=dir+"/"+file_name
    len=64
    img = cv2.imread(file_path)


    img0 = cv2.imread(file_path)
    cv2.imshow('img',img0)
    tgreyim = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(tgreyim)
    print(eyes)
    eye_center = (0,0)

###For Block B cropping
    crop_img = img[0:0+360,0:0+320]
    cv2.imshow('img',crop_img)
    tgreyim = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(tgreyim)
    print(eyes)
    eye_center = (0,0)
    dis=0
    for (s_x, s_y, twid, thei) in eyes:
        Reye_edge_x=s_x+twid
        Reye_edge_y=s_y
        dis=(twid/4)
        Reye_edge = (int(round(Reye_edge_x)), int(round(Reye_edge_y)))
        distance = (thei/2)
        eye_center_x = s_x + (twid/2)
        eye_center_y = s_y + (thei/2)
        eye_center_x = eye_center_x - (twid/2)
        eye_center_y = eye_center_y + distance
        eye_center = (int(round(eye_center_x)), int(round(eye_center_y)))
        print(eye_center_x)
        print(eye_center_y)
        break
    if eye_center == (0,0):
        print("Right Eye not detected, filename " + file_name[:-4])
    else:
        print(eye_center)


    x = eye_center[0]

    if eye_center[1]>479:
        y = 479
        print("blockB y out of boundary, filename "+file_name[:-4])
    else:
        y = eye_center[1]

    blockB = img[y:(y+64),(int(round(x+dis))):(int(round(x+dis))+64)]
    blockB_edge_y = y
    blockB_edge_x = x+64


###For Block D cropping
    crop_img = img[0:0 + 360, 319:319 + 320]
    cv2.imshow('img', crop_img)
    tgreyim = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    Leyes = eye_cascade.detectMultiScale(tgreyim)
    print(Leyes)
    Leye_center = (0, 0)
    dis=0
    for (s_x, s_y, twid, thei) in Leyes:
        Leye_edge_x=s_x+319
        Leye_edge_y=s_y
        dis=int (twid/4)
        Leye_edge = (int(round(Leye_edge_x)), int(round(Leye_edge_y)))
        distance = (thei/2)
        eye_center_x = s_x + (twid / 2)
        eye_center_y = s_y + (thei / 2)
        eye_center_x = eye_center_x + (twid/2) - 64
        eye_center_y = eye_center_y + distance
        Leye_center = (int(round(eye_center_x)), int(round(eye_center_y)))
        print(eye_center_x)
        print(eye_center_y)
        break
    if Leye_center == (0, 0):
        print("Left Eye not detected, filename " + file_name[:-4])
    else:
        print(Leye_center)


    if (Leye_center[0] + 64) > 639:
        x = 575
        print("BlockD - x out of boundary, rectified to 575, file name "+ file_name[:-4])
    else:
        x = Leye_center[0]

    if Leye_center[1] > 479:
        y = 479
        print("blockD y out of boundary, filename " + file_name[:-4])
    else:
        y = Leye_center[1]
    print(str(y)+" is y")
    print(str(x)+" is x")
    blockD = img[y:(y + 64), (x-dis + 319):(x-dis + 319 + 64)]

    blockD_edge_y = y
    blockD_edge_x = x + 319


####Block C
    if (Leye_center != (0,0)) & (eye_center != (0,0)):
        blockB_edge = (blockB_edge_x, blockB_edge_y)
        blockD_edge = (blockD_edge_x, blockD_edge_y)
        x_diff=blockD_edge[0]-blockB_edge[0]
        if x_diff < 64:
            print("Cropping of block B and D not successful, filename "+file_name[:-4])
            continue
    else:
        print("Please do cropping manually, filename "+file_name[:-4])
        continue

    mid_x=(x_diff/2)
    mid_x=blockB_edge[0]+mid_x
    x = int (mid_x - 32)
    y = int ((blockB_edge[1]+blockD_edge[1])/2)
    y = y - 28
    blockC = img[y:(y+64), x:(x+64)]


####Block A
    xdiff=Reye_edge[0] - Leye_edge[0]
    mid_x=(xdiff/2)
    mid_x=Leye_edge[0]+mid_x
    x = int (mid_x - 32)
    y = int ((Leye_edge[1]+Reye_edge[1])/2)
    y = y - 65
    if y - 65 <0:
        y=0
        print("blockA - y out of boundary. rectified to y=0")
    blockA = img[y:(y+64), x:(x+64)]


    blockApath="D:/FYP/face_blocks/"+type+"/"+file_name[:-4]+"_A.bmp"
    cv2.imwrite(blockApath,blockA)

    blockBpath="D:/FYP/face_blocks/"+type+"/"+file_name[:-4]+"_B.bmp"
    cv2.imwrite(blockBpath,blockB)

    blockCpath="D:/FYP/face_blocks/"+type+"/"+file_name[:-4]+"_C.bmp"
    cv2.imwrite(blockCpath,blockC)

    blockDpath="D:/FYP/face_blocks/"+type+"/"+file_name[:-4]+"_D.bmp"
    cv2.imwrite(blockDpath,blockD)

    cv2.imwrite("D:/demo/"+file_name[:-4]+".bmp",img0)
    break
