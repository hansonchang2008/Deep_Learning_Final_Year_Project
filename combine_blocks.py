from PIL import Image
import os
import sys


def combine_images(patient_id,image_path_face,image_path_tongue,image_path_output):
    #merge horizontally
    file_face_A=image_path_face+patient_id+"_A.bmp"
    file_face_B=image_path_face+patient_id+"_B.bmp"
    file_face_C=image_path_face+patient_id+"_C.bmp"
    file_face_D=image_path_face+patient_id+"_D.bmp"
    imA=Image.open(file_face_A)
    imB=Image.open(file_face_B)
    imC=Image.open(file_face_C)
    imD=Image.open(file_face_D)
    images=[imA, imB, imC, imD]
    width = 256
    height = 64
    new_im = Image.new('RGB', (width, height)) #Create new image
    i=0
    x_offset = 0
    while i != 4:
       new_im.paste(images[i], (x_offset,0))
       x_offset = x_offset+64
       i=i+1
    part1=image_path_output+patient_id+"_F.bmp"
    new_im.save(part1)
    
    #merge horizontally
    file_tongue_1=image_path_tongue+patient_id+"-1.bmp"
    file_tongue_2=image_path_tongue+patient_id+"-2.bmp"
    file_tongue_3=image_path_tongue+patient_id+"-3.bmp"
    file_tongue_4=image_path_tongue+patient_id+"-4.bmp"
    im1=Image.open(file_tongue_1)
    im2=Image.open(file_tongue_2)
    im3=Image.open(file_tongue_3)
    im4=Image.open(file_tongue_4)
    images=[im1, im2, im3, im4]
    width = 256
    height = 64
    new_im = Image.new('RGB', (width, height)) #Create new image
    i=0
    x_offset = 0
    while i != 4:
       new_im.paste(images[i], (x_offset,0))
       x_offset = x_offset+64
       i=i+1
    part2=image_path_output+patient_id+"_T1.bmp"
    new_im.save(part2)
    
    #merge horizontally
    file_tongue_5=image_path_tongue+patient_id+"-5.bmp"
    file_tongue_6=image_path_tongue+patient_id+"-6.bmp"
    file_tongue_7=image_path_tongue+patient_id+"-7.bmp"
    file_tongue_8=image_path_tongue+patient_id+"-8.bmp"
    im5=Image.open(file_tongue_5)
    im6=Image.open(file_tongue_6)
    im7=Image.open(file_tongue_7)
    im8=Image.open(file_tongue_8)
    images=[im5, im6, im7, im8]
    width = 256
    height = 64
    new_im = Image.new('RGB', (width, height)) #Create new image
    i=0
    x_offset = 0
    while i != 4:
       new_im.paste(images[i], (x_offset,0))
       x_offset = x_offset+64
       i=i+1
    part3=image_path_output+patient_id+"_T2.bmp"
    new_im.save(part3)
    
    #Merge part1 part2 part3 vertically
    P1=Image.open(part1)
    P2=Image.open(part2)
    P3=Image.open(part3)
    images = [P1, P2, P3]
    width = 256
    height = 192
    new_im = Image.new('RGB', (width, height))
    i=0
    y_offset = 0
    while i != 3:
       new_im.paste(images[i], (0,y_offset))
       y_offset = y_offset+64
       i=i+1
    combined_image=image_path_output+patient_id+".bmp"
    new_im.save(combined_image)
    #remove the three parts
    if os.path.exists(part1):
        os.remove(part1)
    else:
        print("file noexist"+" "+part1)
    if os.path.exists(part2):
        os.remove(part2)
    else:
        print("file noexist"+" "+part2)
    if os.path.exists(part3):
        os.remove(part3)
    else:
        print("file noexist"+" "+part3)
    return True

def check_and_combine(image_path_face,image_path_tongue,image_path_output):
    #Iterate over each file in the face directory
    for filename in os.listdir(image_path_face):
        print(filename)
        filename1=os.path.splitext(filename)[0]
        print(filename1)
        if filename1.endswith('A'):
            patient_id=filename1[:9] #Get the first 9 characters of the image file name
            print(patient_id)
            #Find whether the tongue images of the patient exist
            path_tongue=image_path_tongue+patient_id+"-1.bmp"
            print(path_tongue)
            path_tongue_folder=image_path_tongue
            if os.path.exists(path_tongue):
            #Tongue images of the patient exist, combine the face and tongue images
                print("Both face and tongue image of "+patient_id+"exists, combine them")
                #Combine the images
                flag=combine_images(patient_id,image_path_face,image_path_tongue,image_path_output)
                if flag is True:
                    continue
                else:
                    print("Error occurs"+" "+patient_id+" "+classname)
            else:
                print("tongue image not found for"+" "+patient_id)
                continue
        else:
            continue
    return True



#main function, check and merge the images of each patient into one image
#
#Get Current Path of the python script file
current_path= os.path.dirname(os.path.abspath(__file__))
print(current_path)

#healthy class
classname="healthy"
image_path_face = current_path+"\\face data\\"+classname+"\\" #Path to the input images
image_path_tongue = current_path+"\\tongue data\\"+classname+"\\" #Path to the input images
image_path_output = current_path+"\\combined_images\\"+classname+"\\" #path to the output folder
flag=check_and_combine(image_path_face,image_path_tongue,image_path_output) #Check data completeness and combine images
print(classname+" "+str(flag))

#diabetes class
classname="diabetes"
image_path_face = current_path+"\\face data\\"+classname+"\\"
image_path_tongue = current_path+"\\tongue data\\"+classname+"\\"
image_path_output = current_path+"\\combined_images\\"+classname+"\\" #path to the output folder
check_and_combine(image_path_face,image_path_tongue,image_path_output) #Check data completeness and combine images
print(classname+" "+str(flag))

#heart disease class
classname="heart disease"
image_path_face = current_path+"\\face data\\"+classname+"\\"
image_path_tongue = current_path+"\\tongue data\\"+classname+"\\"
image_path_output = current_path+"\\combined_images\\"+classname+"\\" #path to the output folder
check_and_combine(image_path_face,image_path_tongue,image_path_output) #Check data completeness and combine images
print(classname+" "+str(flag))
