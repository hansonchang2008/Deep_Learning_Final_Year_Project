import os
import math
from PIL import Image

type="healthy"
dir="D:/FYP/whole_tongue/"+str(type)
for file_name in os.listdir(dir):
    print(file_name)
    file_path=dir+"/"+file_name
    img = Image.open(file_path)
    print(img)
    len=64
    im01=img
    width, height = im01.size
    im01x_rate=0.3985
    im01y_rate=0.7552
    im01x=width*im01x_rate
    im01y=height*im01y_rate
    im01xe=im01x+len
    im01ye=im01y+len
    seg01=(im01x,im01y,im01xe,im01ye)
    block1=im01.crop(seg01)
  #  img.show()

    #2nd block
    width_o, height_o =img.size
    im02=img.transpose(Image.FLIP_LEFT_RIGHT)
  #  im02.show()
    #255 rotation rate
    im02 =im02.convert('RGBA')

    im02t =im02.rotate(105, expand=1)

    fff = Image.new('RGBA', im02t.size, (255, 255, 255, 255))

    im02t = Image.composite(im02t,fff,im02t)
    im02t=im02t.convert(img.mode)

    #im02t.show()
    #im02.save("D:/rotated01.bmp")
    im02t.save("D:/flipped00Compare_Flip_thenROTATE.bmp")
    width, height = im02t.size

    rate_w = 0.0448875107
    rate_h = 0.5585375325

    #m = (width_o * rate_w) * math.cos(math.radians(15))
    #n = height_o * math.sin(math.radians(15))
    #im02y = n + m
    im02x = (height_o * rate_h) * math.cos(math.radians(15))
    yo = width_o * rate_w
    print("yo"+str(yo))
    y2 = yo/math.cos(math.radians(15))
    print("y2"+str(y2))
    ny = im02x * math.tan(math.radians(15))
    print("ny"+str(ny))
    my = ny - y2
    print("my"+str(my))
    im02y = height_o * math.sin(math.radians(15)) - my
    print(im02x, im02y)
    #im02x_rate = 0.4483
    #im02y_rate = 0.1380
    #im02x = width * im02x_rate
    #print("wid"+str(im02x))
    #he= 156.45669 / math.cos(math.radians(15))
    #print("original height"+str(he))
    #rate_h = he/height_o
    #im02y = height * im02y_rate
    #print("y"+str(im02y))
    #print("height"+str(height))
    #my = height_o * math.sin(math.radians(15))-im02y
    #print("my"+str(my))
    #ny = im02x * math.tan(math.radians(15))
    #print("ny"+str(ny))
#    y2 = ny - my
#   yo = y2 * math.cos(math.radians(15))
#    rate_w=yo/width_o
#    print("original y is "+str(yo))
#    rate_w = 1 - rate_w
#    print("rate_w"+str(rate_w))



    im02xe = im02x+len
    im02ye = im02y+len
    seg02=(im02x,im02y,im02xe,im02ye)
    block2=im02t.crop(seg02)

   # block2.show()


    #3rd block
    im03t=im02t
    #im03t.show()
    rate_w = 0.01532567049
    rate_h = 0.23448275862
    im03x = (height_o * rate_h) * math.cos(math.radians(15))
    yo = width_o * rate_w
    #print("yo"+str(yo))
    y2 = yo/math.cos(math.radians(15))
    #print("y2"+str(y2))
    ny = im03x * math.tan(math.radians(15))
    #print("ny"+str(ny))
    my = ny - y2
    #print("my"+str(my))
    im03y = height_o * math.sin(math.radians(15)) - my
    #print(im03x, im03y)

    #im03x_rate = 0.1925
    #im03y_rate = 0.1865

    #im03x = width * im03x_rate
    #im03y = height * im03y_rate

    im03xe = im03x+len
    im03ye = im03y+len
    seg03=(im03x,im03y,im03xe,im03ye)
    block3=im03t.crop(seg03)
    #block3.show()

    #4th block
    #255 rotation rate
    img =img.convert('RGBA')
    im04 = img.rotate(110, expand=1)

    fff = Image.new('RGBA', im04.size, (255, 255, 255, 255))

    im04 = Image.composite(im04,fff,im04)
    im04=im04.convert(img.mode)



    im04.save("D:/4.bmp")
    #im04.show()
    width, height = im04.size
    rate_w = 0.06213409962
    rate_h = 0.57620689655

    im04x = (height_o * rate_h) * math.cos(math.radians(20))
    yo = width_o * rate_w
    # print("yo"+str(yo))
    y2 = yo / math.cos(math.radians(20))
    # print("y2"+str(y2))
    ny = im04x * math.tan(math.radians(20))
    # print("ny"+str(ny))
    my = ny - y2
    # print("my"+str(my))
    im04y = height_o * math.sin(math.radians(20)) - my
    # print(im03x, im03y)

    #im04x_rate=0.4504
    #im04y_rate=0.1705
    #im04x = width * im04x_rate
    #im04y = height * im04y_rate




    im04xe=im04x+len
    im04ye=im04y+len
    seg04=(im04x,im04y,im04xe,im04ye)
    block4=im04.crop(seg04)
   # block4.show()




    #5th block
   # img.show()
    img =img.convert('RGBA')
    im05 = img.rotate(100, expand=1)


    fff = Image.new('RGBA', im05.size, (255, 255, 255, 255))
    im05 = Image.composite(im05,fff,im05)
    im05=im05.convert(img.mode)
    #im05.show()
    im05.save("D:/rotated5.bmp")
    width, height = im05.size

    rate_w = 1 - 0.99233716475
    rate_h = 0.22758620689

    im05x = (height_o * rate_h) * math.cos(math.radians(10))
    yo = width_o * rate_w
    # print("yo"+str(yo))
    y2 = yo / math.cos(math.radians(10))
    # print("y2"+str(y2))
    ny = im05x * math.tan(math.radians(10))
    # print("ny"+str(ny))
    my = ny - y2
    # print("my"+str(my))
    im05y = height_o * math.sin(math.radians(10)) - my
    # print(im03x, im03y)
#    im05x_rate = 0.1964
#   im05y_rate = 0.1364
#    im05x = width * im05x_rate
#    im05y = width * im05y_rate
    im05xe = im05x + len
    im05ye = im05y + len
    seg05=(im05x, im05y, im05xe, im05ye)
    block5=im05.crop(seg05)
   # block5.show()

    #64 28
    #261 290
    im06=img
    width, height = im06.size
    im06x_rate=0.2452
    im06y_rate=0.0956
    im06x = width * im06x_rate
    im06y = height *im06y_rate
    im06xe = im06x + len
    im06ye = im06y +len
    seg06 = (im06x, im06y, im06xe, im06ye)
    block6 = im06.crop(seg06)
    #128 27
    #261 290
    im07=img
    width, height = im07.size
    im07x_rate=0.4904
    im07y_rate=0.0931
    im07x = width * im07x_rate
    im07y = width * im07y_rate
    im07xe = im07x + len
    im07ye = im07y + len
    seg07 = (im07x, im07y, im07xe, im07ye)
    block7 = im07.crop(seg07)

    #96 90
    #261 290
    im08=img
    width, height = im08.size
    im08x_rate = 0.3678
    im08y_rate = 0.3103
    im08x = width * im08x_rate
    im08y = width * im08y_rate
    im08xe = im08x +len
    im08ye = im08y +len
    seg08 = (im08x, im08y, im08xe, im08ye)
    block8 = im08.crop(seg08)

    block1path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-1.bmp"
    block1.save(block1path)
    block2path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-2.bmp"
    block2.save(block2path)
    block3path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-3.bmp"
    block3.save(block3path)
    block4path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-4.bmp"
    block4.save(block4path)
    block5path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-5.bmp"
    block5.save(block5path)
    block6path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-6.bmp"
    block6.save(block6path)
    block7path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-7.bmp"
    block7.save(block7path)
    block8path="D:/FYP/tongue_blocks/"+type+"/"+file_name[:-4]+"-8.bmp"
    block8.save(block8path)

#break;