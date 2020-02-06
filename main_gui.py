# import all modules
from PyQt5.QtGui import *
import cv2
from PIL import Image
import math
import shutil
from my_gui_change_backgroud import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication,QVBoxLayout
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import sys
import demo_testing

current_dir = r'%s' % os.getcwd().replace('\\','/')
os.chdir(current_dir)
sys.path.append(current_dir)
if not os.path.exists('images'):
    os.mkdir('images')

eye_cascade = cv2.CascadeClassifier(current_dir+'/haarcascade_eye.xml')


class MyMainGui(QMainWindow, Ui_MainWindow):
    def face_cropping(self):
        file_name=self.face_file_path
        print('Inside the face_cropping function'+file_name)
        img = cv2.imread(file_name)
        ##show the result of eye detection
        img0 = cv2.imread(file_name)
        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        ###For Block B cropping
        crop_img = img[0:0 + 360, 0:0 + 320]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray)
        eye_center = (0, 0)
        dis = 0
        for (ex, ey, ew, eh) in eyes:
            Reye_edge_x = ex + ew
            Reye_edge_y = ey
            dis = (ew / 4)
            Reye_edge = (int(round(Reye_edge_x)), int(round(Reye_edge_y)))
            distance = (eh / 2)
            eye_center_x = ex + (ew / 2)
            eye_center_y = ey + (eh / 2)
            eye_center_x = eye_center_x - (ew / 2)
            eye_center_y = eye_center_y + distance
            eye_center = (int(round(eye_center_x)), int(round(eye_center_y)))
            break
        if eye_center == (0, 0):
            print("Right Eye not detected, filename " + file_name[:-4])
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        x = eye_center[0]

        if eye_center[1] > 479:
            y = 479
            print("blockB y out of boundary, filename " + file_name[:-4])
        else:
            y = eye_center[1]

        blockB = img[y:(y + 64), (int(round(x + dis))):(int(round(x + dis)) + 64)]
        blockB_edge_y = y
        blockB_edge_x = x + 64

        ###For Block D cropping
        crop_img = img[0:0 + 360, 319:319 + 320]
        #cv2.imshow('img', crop_img)
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        Leyes = eye_cascade.detectMultiScale(gray)
        #print(Leyes)
        Leye_center = (0, 0)
        dis = 0
        for (ex, ey, ew, eh) in Leyes:
            #   draw rectangle around the eye:
            Leye_edge_x = ex + 319
            Leye_edge_y = ey
            dis = int(ew / 4)
            Leye_edge = (int(round(Leye_edge_x)), int(round(Leye_edge_y)))
            distance = (eh / 2)
            eye_center_x = ex + (ew / 2)
            eye_center_y = ey + (eh / 2)
            eye_center_x = eye_center_x + (ew / 2) - 64
            eye_center_y = eye_center_y + distance
            Leye_center = (int(round(eye_center_x)), int(round(eye_center_y)))
            break
        if Leye_center == (0, 0):
            print("Left Eye not detected, filename " + file_name[:-4])
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if (Leye_center[0] + 64) > 639:
            x = 575
            print("BlockD - x out of boundary, rectified to 575, file name " + file_name[:-4])
        else:
            x = Leye_center[0]

        if Leye_center[1] > 479:
            y = 479
            print("blockD y out of boundary, filename " + file_name[:-4])
        else:
            y = Leye_center[1]
        blockD = img[y:(y + 64), (x - dis + 319):(x - dis + 319 + 64)]

        blockD_edge_y = y
        blockD_edge_x = x + 319

        ####Block C
        if (Leye_center != (0, 0)) & (eye_center != (0, 0)):
            blockB_edge = (blockB_edge_x, blockB_edge_y)
            blockD_edge = (blockD_edge_x, blockD_edge_y)
            x_diff = blockD_edge[0] - blockB_edge[0]
            if x_diff < 64:
                print("Cropping of block B and D not successful, filename " + file_name[:-4])
        else:
            print("Please do cropping manually, filename " + file_name[:-4])

        mid_x = (x_diff / 2)
        mid_x = blockB_edge[0] + mid_x
        x = int(mid_x - 32)
        y = int((blockB_edge[1] + blockD_edge[1]) / 2)
        y = y - 28
        blockC = img[y:(y + 64), x:(x + 64)]

        ####Block A
        xdiff = Reye_edge[0] - Leye_edge[0]
        mid_x = (xdiff / 2)
        mid_x = Leye_edge[0] + mid_x
        x = int(mid_x - 32)
        y = int((Leye_edge[1] + Reye_edge[1]) / 2)
        y = y - 65
        if y - 65 < 0:
            y = 0
            print("blockA - y out of boundary. rectified to y=0")
        blockA = img[y:(y + 64), x:(x + 64)]

        file_name_base=os.path.basename(file_name)

        blockApath = current_dir + "/images/" + file_name_base[:-4] + "_A.bmp"
        cv2.imwrite(blockApath, blockA)

        blockBpath = current_dir + "/images/" + file_name_base[:-4] + "_B.bmp"
        cv2.imwrite(blockBpath, blockB)

        blockCpath = current_dir + "/images/" + file_name_base[:-4] + "_C.bmp"
        cv2.imwrite(blockCpath, blockC)

        blockDpath = current_dir + "/images/" + file_name_base[:-4] + "_D.bmp"
        cv2.imwrite(blockDpath, blockD)


    def tongue_cropping_rec(self):
        file_path=self.tongue_file_path
        print(file_path)
        file_name=os.path.basename(file_path)
        print(file_name)
        img = Image.open(file_path)
        print(img)
        len = 64
        im01 = img
        width, height = im01.size
        im01x_rate = 0.3985
        im01y_rate = 0.7422
        im01x = width * im01x_rate
        im01y = height * im01y_rate
        im01xe = im01x + len
        im01ye = im01y + len
        seg01 = (im01x, im01y, im01xe, im01ye)
        block1 = im01.crop(seg01)

        # 2nd block
        width_o, height_o = img.size
        im02 = img.transpose(Image.FLIP_LEFT_RIGHT)
        im02 = im02.convert('RGBA')

        im02t = im02.rotate(105, expand=1)
        fff = Image.new('RGBA', im02t.size, (255, 255, 255, 255))
        im02t = Image.composite(im02t, fff, im02t)
        im02t = im02t.convert(img.mode)


        rate_w = 0.0808875107
        rate_h = 0.5385375325

        im02x = (height_o * rate_h) * math.cos(math.radians(15))
        yo = width_o * rate_w
        print("yo" + str(yo))
        y2 = yo / math.cos(math.radians(15))
        print("y2" + str(y2))
        ny = im02x * math.tan(math.radians(15))
        print("ny" + str(ny))
        my = ny - y2
        print("my" + str(my))
        im02y = height_o * math.sin(math.radians(15)) - my
        print(im02x, im02y)
        im02xe = im02x + len
        im02ye = im02y + len
        seg02 = (im02x, im02y, im02xe, im02ye)
        block2 = im02t.crop(seg02)

        # 3rd block
        im03t = im02t
        rate_w = 0.02032567049
        rate_h = 0.23448275862
        im03x = (height_o * rate_h) * math.cos(math.radians(15))
        yo = width_o * rate_w
        y2 = yo / math.cos(math.radians(15))
        ny = im03x * math.tan(math.radians(15))
        my = ny - y2
        im03y = height_o * math.sin(math.radians(15)) - my
        im03xe = im03x + len
        im03ye = im03y + len
        seg03 = (im03x, im03y, im03xe, im03ye)
        block3 = im03t.crop(seg03)

        # 4th block
        img = img.convert('RGBA')
        im04 = img.rotate(110, expand=1)
        fff = Image.new('RGBA', im04.size, (255, 255, 255, 255))
        im04 = Image.composite(im04, fff, im04)
        im04 = im04.convert(img.mode)

        rate_w = 0.06213409962
        rate_h = 0.57620689655

        im04x = (height_o * rate_h) * math.cos(math.radians(20))
        yo = width_o * rate_w
        y2 = yo / math.cos(math.radians(20))
        ny = im04x * math.tan(math.radians(20))
        my = ny - y2
        im04y = height_o * math.sin(math.radians(20)) - my

        im04xe = im04x + len
        im04ye = im04y + len
        seg04 = (im04x, im04y, im04xe, im04ye)
        block4 = im04.crop(seg04)

        # 5th block
        img = img.convert('RGBA')
        im05 = img.rotate(100, expand=1)

        # 与旋转图像大小相同的白色图像
        fff = Image.new('RGBA', im05.size, (255, 255, 255, 255))
        # 使用alpha层的rot作为掩码创建一个复合图像
        im05 = Image.composite(im05, fff, im05)
        im05 = im05.convert(img.mode)

        rate_w = 1 - 0.97033716475
        rate_h = 0.22758620689

        im05x = (height_o * rate_h) * math.cos(math.radians(10))
        yo = width_o * rate_w
        y2 = yo / math.cos(math.radians(10))
        ny = im05x * math.tan(math.radians(10))
        my = ny - y2
        im05y = height_o * math.sin(math.radians(10)) - my
        im05xe = im05x + len
        im05ye = im05y + len
        seg05 = (im05x, im05y, im05xe, im05ye)
        block5 = im05.crop(seg05)

        im06 = img
        width, height = im06.size
        im06x_rate = 0.2452
        im06y_rate = 0.0956
        im06x = width * im06x_rate
        im06y = height * im06y_rate
        im06xe = im06x + len
        im06ye = im06y + len
        seg06 = (im06x, im06y, im06xe, im06ye)
        block6 = im06.crop(seg06)
        im07 = img
        width, height = im07.size
        im07x_rate = 0.4904
        im07y_rate = 0.0931
        im07x = width * im07x_rate
        im07y = width * im07y_rate
        im07xe = im07x + len
        im07ye = im07y + len
        seg07 = (im07x, im07y, im07xe, im07ye)
        block7 = im07.crop(seg07)

        im08 = img
        width, height = im08.size
        im08x_rate = 0.3678
        im08y_rate = 0.3103
        im08x = width * im08x_rate
        im08y = width * im08y_rate
        im08xe = im08x + len
        im08ye = im08y + len
        seg08 = (im08x, im08y, im08xe, im08ye)
        block8 = im08.crop(seg08)

        block1path = current_dir + "/images/" + file_name[:-4] + "-1.bmp"
        block1.save(block1path)
        block2path = current_dir + "/images/" + file_name[:-4] + "-2.bmp"
        block2.save(block2path)
        block3path = current_dir + "/images/" + file_name[:-4] + "-3.bmp"
        block3.save(block3path)
        block4path = current_dir + "/images/" + file_name[:-4] + "-4.bmp"
        block4.save(block4path)
        block5path = current_dir + "/images/" + file_name[:-4] + "-5.bmp"
        block5.save(block5path)
        block6path = current_dir + "/images/" + file_name[:-4] + "-6.bmp"
        block6.save(block6path)
        block7path = current_dir + "/images/" + file_name[:-4] + "-7.bmp"
        block7.save(block7path)
        block8path = current_dir + "/images/" + file_name[:-4] + "-8.bmp"
        block8.save(block8path)

    def combine_images(self):
        face_name=os.path.basename(self.face_file_path)
        tongue_name=os.path.basename(self.tongue_file_path)
        print(face_name)
        print(tongue_name)
        imA = Image.open(current_dir+"/images/"+face_name[:-4]+"_A.bmp")
        imB = Image.open(current_dir+"/images/"+face_name[:-4]+"_B.bmp")
        imC = Image.open(current_dir+"/images/"+face_name[:-4]+"_C.bmp")
        imD = Image.open(current_dir+"/images/"+face_name[:-4]+"_D.bmp")
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
        part1=current_dir+"/images/"+face_name[:-4]+"_F.bmp"
        new_im.save(part1)

        # merge horizontally
        file_tongue_1 = current_dir+"/images/"+tongue_name[:-4] + "-1.bmp"
        file_tongue_2 = current_dir+"/images/"+tongue_name[:-4] + "-2.bmp"
        file_tongue_3 = current_dir+"/images/"+tongue_name[:-4] + "-3.bmp"
        file_tongue_4 = current_dir+"/images/"+tongue_name[:-4] + "-4.bmp"
        im1 = Image.open(file_tongue_1)
        im2 = Image.open(file_tongue_2)
        im3 = Image.open(file_tongue_3)
        im4 = Image.open(file_tongue_4)
        images = [im1, im2, im3, im4]
        width = 256
        height = 64
        new_im = Image.new('RGB', (width, height))  # Create new image
        i = 0
        x_offset = 0
        while i != 4:
            new_im.paste(images[i], (x_offset, 0))
            x_offset = x_offset + 64
            i = i + 1
        part2 = current_dir+"/images/"+tongue_name[:-4] + "_T1.bmp"
        new_im.save(part2)

        # merge horizontally
        file_tongue_5 = current_dir+"/images/"+tongue_name[:-4]+ "-5.bmp"
        file_tongue_6 = current_dir+"/images/"+tongue_name[:-4]+ "-6.bmp"
        file_tongue_7 = current_dir+"/images/"+tongue_name[:-4]+ "-7.bmp"
        file_tongue_8 = current_dir+"/images/"+tongue_name[:-4]+ "-8.bmp"
        im5 = Image.open(file_tongue_5)
        im6 = Image.open(file_tongue_6)
        im7 = Image.open(file_tongue_7)
        im8 = Image.open(file_tongue_8)
        images = [im5, im6, im7, im8]
        width = 256
        height = 64
        new_im = Image.new('RGB', (width, height))  # Create new image
        i = 0
        x_offset = 0
        while i != 4:
            new_im.paste(images[i], (x_offset, 0))
            x_offset = x_offset + 64
            i = i + 1
        part3 = current_dir+"/images/"+tongue_name[:-4]+ "_T2.bmp"
        new_im.save(part3)

        # Merge part1 part2 part3 vertically
        P1 = Image.open(part1)
        P2 = Image.open(part2)
        P3 = Image.open(part3)
        images = [P1, P2, P3]
        width = 256
        height = 192
        new_im = Image.new('RGB', (width, height))
        i = 0
        y_offset = 0
        while i != 3:
            new_im.paste(images[i], (0, y_offset))
            y_offset = y_offset + 64
            i = i + 1
        combined_image = current_dir+"/images/"+face_name[:-4] + ".bmp"
        new_im.save(combined_image)
        # remove the three parts
        if os.path.exists(part1):
            os.remove(part1)
        else:
            print("The file does not exist" + " " + part1)
        if os.path.exists(part2):
            os.remove(part2)
        else:
            print("The file does not exist" + " " + part2)
        if os.path.exists(part3):
            os.remove(part3)
        else:
            print("The file does not exist" + " " + part3)
        return combined_image



    #extend constructors of QMainWindow and Ui_mainWindow
    def __init__(self, parent=None):
        super(MyMainGui, self).__init__()
        # call setupUi function in Ui_MainWindow class
        self.setupUi(self)
        self.consoleText = 'Application started'
        self.file_path_face.setWordWrap(True)
        self.file_path_tongue.setWordWrap(True)


        self.logo_file_path=current_dir+"/logo.png"
        self.logo_img = QPixmap(self.logo_file_path)
        self.label.setPixmap(self.logo_img.scaled(240, 70))

        self.selectFace.clicked.connect(self.show_face_image)
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.faceImage)
        self.vlayout.addWidget(self.selectFace)
        self.setLayout(self.vlayout)
        self.clearFace.clicked.connect(self.clear_face)

        self.selectTongue.clicked.connect(self.show_tongue_image)
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.tongueImage)
        self.vlayout.addWidget(self.selectTongue)
        self.setLayout(self.vlayout)
        self.clearTongue.clicked.connect(self.clear_tongue)

        self.checkHealthStatus.clicked.connect(self.check_health_status)

    def show_face_image(self):
        self.face_file_path = QFileDialog.getOpenFileName(self, "OpenFile", ".", "Image Files(*.bmp)")[0]
        if len(self.face_file_path):
            self.face_img = QPixmap(self.face_file_path)
            h = self.faceImage.height()
            w = self.faceImage.width()
            self.faceImage.setPixmap(self.face_img.scaled(w,h))
            self.file_path_face.setText('Face Image Path: '+self.face_file_path)
            print('facefilepath: ' + str(self.face_file_path))

    def clear_face(self):
        self.faceImage.setPixmap(QPixmap(""))
        self.faceImage.setText("Face Image")
        self.face_file_path = ""
        self.file_path_face.setText("Face image file path")
        print('facefilepath: '+str(self.face_file_path))

    def show_tongue_image(self):
        self.tongue_file_path = QFileDialog.getOpenFileName(self, "OpenFile", ".", "Image Files(*.bmp)")[0]
        if len(self.tongue_file_path):
            self.tongue_img = QPixmap(self.tongue_file_path)
            h = self.tongueImage.height()
            w = self.tongueImage.width()
            self.tongueImage.setPixmap(self.tongue_img.scaled(w,h))
            self.file_path_tongue.setText('Tongue Image Path: '+self.tongue_file_path)
            print('tonguefilepath: ' + str(self.tongue_file_path))

    def clear_tongue(self):
        self.tongueImage.setPixmap(QPixmap(""))
        self.tongueImage.setText("Tongue Image")
        self.tongue_file_path = ""
        self.file_path_tongue.setText("Tongue image file path")
        print('tonguefilepath: '+str(self.tongue_file_path))

    def check_health_status(self):
        print(self.face_file_path)
        print(self.tongue_file_path)

        try:
            self.face_cropping()
        except:
            print("Error occurred calling face_cropping")

        tongueShape = "Rectangle"
        print(tongueShape)
        if tongueShape == "Rectangle":
            try:
                self.tongue_cropping_rec()
            except:
                print("Error occured calling tongue_cropping_rec")
        elif tongueShape == "Triangle":
            try:
                self.tongue_cropping_tri()
            except:
                print("Error occured calling tongue_cropping_tri")
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Other shape not supported yet.')
            msg.setWindowTitle("Error")
            msg.exec_()
            print("Other shape not supported yet.")
        try:
            combined_image=self.combine_images()
        except:
            print("Error during combining images")
        print("combined_image is path "+combined_image)
        label, pos = demo_testing.test(combined_image)
        label=int(label)
        pos=float(pos)
        print("labellll"+str(label))
        pos_percent = ('{:.4%}'.format(pos))

        print(pos_percent)
        if label == 0:
            print("healthy %.4f"%pos)
            result = "There is a " +str(pos_percent)+" chance that you are healthy."
        elif label == 1:
            print("diabetes %.4f"%pos)
            result = "<p><font size =\"4\">There is a <strong>"+str(pos_percent)+"</strong> chance that you are suffering from <font color=\"red\">diabetes</font>.</font></p>"+\
                     "\n<p><font size = \"4\">Please seek medical assistance at your earliest convenience.</font></p>"

        elif label == 2:
            print("heart disease %.4f"%pos)
            result = "<p><font size =\"4\">There is a <strong>"+str(pos_percent)+"</strong> chance that you are suffering from <font color=\"red\">heart disease</font>.</font></p>"+\
                     "\n<p><font size = \"4\">Please seek medical assistance at your earliest convenience.</font></p>"

        else:
            print("something goes wrong on prediction.")
            result = "Error occurs."
        msg=QMessageBox.about(self, "Multi-disease Detection Result", str(result))




    def closeEvent(self, event):
        shutil.rmtree(current_dir + "/images")
        event.accept() #close the window





if __name__ == "__main__":
    app = QApplication([])
    my_gui = MyMainGui()
    my_gui.show()
    app.exit(app.exec_())

