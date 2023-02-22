import tkinter as tk
import cv2
import numpy as np
import pymysql as mysql
from PIL import Image,ImageTk
import cpSeg.segment as seg
import getCpNum as cp
pading=20
global hist
def getBinary_pic(path):
    """
    :param path:pic_path
    :return: binary_pic
    """
    img=cv2.imread(path,0)
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    index = np.argmax(hist[100:255])
    print(f"maxPix={100+index}")
    ret, img2 = cv2.threshold(img, (100+index), 255, cv2.THRESH_BINARY)
    return img2

def getCp(img):
    nPlateCascade = cv2.CascadeClassifier("./Resource/haarcascade_russian_plate_number.xml")
    numberPlates = nPlateCascade.detectMultiScale(img, 1.1, 10)
    len_numPlate= len(numberPlates)
    if len_numPlate==0:
        img = cv2.resize(img, (640, 200))
        return 0,img
    else:
        (x, y, w, h)=numberPlates[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        imgRoi = img[y:y + h, x-pading:x + w+pading]
        imgRoi = cv2.resize(imgRoi, (640, 200))
        return 1,imgRoi

def getImg():
    path_now=e.get()
    seg.main_cp(path_now)
    global photo_now
    photo_now=ImageTk.PhotoImage(Image.open(path_now).resize((400,200)))
    phote_label.config(image=photo_now)
    # ret,cp=getCp(cv2.imread(path_now))
    # cv2.imwrite("E:/photo/run/cp.jpg",cp)
    #cp_now改为二值图像
    global cp_now
    cp_now=ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/binary_img.bmp"))
    cp_label.config(image=cp_now)
    # cp_binary=getBinary_pic("E:/photo/run/cp.jpg")
    # cv2.imwrite("E:/photo/run/cp_binary.jpg",cp_binary)
    #cp_binary_img改为分割的图像
    global cp_binary_img_0
    cp_binary_img_0 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/0.bmp"))
    binary_label_0.config(image=cp_binary_img_0)
    global cp_binary_img_1
    cp_binary_img_1 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/1.bmp"))
    binary_label_1.config(image=cp_binary_img_1)
    global cp_binary_img_2
    cp_binary_img_2 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/2.bmp"))
    binary_label_2.config(image=cp_binary_img_2)
    global cp_binary_img_3
    cp_binary_img_3 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/3.bmp"))
    binary_label_3.config(image=cp_binary_img_3)
    global cp_binary_img_4
    cp_binary_img_4 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/4.bmp"))
    binary_label_4.config(image=cp_binary_img_4)
    global cp_binary_img_5
    cp_binary_img_5 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/5.bmp"))
    binary_label_5.config(image=cp_binary_img_5)
    global cp_binary_img_6
    cp_binary_img_6 = ImageTk.PhotoImage(Image.open("E:/pictures/cp/chars/6.bmp"))
    binary_label_6.config(image=cp_binary_img_6)

    cp_str=cp.getCpNum()
    str_Entry.delete(0,"end")
    str_Entry.insert(0,cp_str)


def getInfo():
    cp_num=str_Entry.get()
    ret = cursor.execute(f"select * from time_info where id='{cp_num}'")
    if(ret==0):print("查无此车牌信息")
    row: tuple = cursor.fetchone()
    inTime_Entry.insert(0,row[1])
    outTime_Entry.insert(0,row[2])
    total_Time_label.config(text="停车时长"+str(row[2]-row[1]))

def inAction():
    cp_num = str_Entry.get()
    ret = cursor.execute(f"insert into time_info(id,inTime,outTime)values('{cp_num}',NOW(),NULL)")
    if(ret>0):total_Time_label.config(text="已记录停车时间")
    else:total_Time_label.config(text="停车时间出错")

def outAction():
    cp_num = str_Entry.get()
    ret = cursor.execute(f"update time_info set outTime=NOW() where id='{cp_num}'")
    if (ret > 0):
        total_Time_label.config(text="已记录离开时间")
    else:
        total_Time_label.config(text="离开时间出错")


import pymysql as mysql
conn=mysql.Connection(host="localhost",user="root",password="xfblackzero20012",database="car_stop_place",port=3306)
cursor=conn.cursor()
conn.select_db("car_stop_place")
conn.autocommit(True)


window=tk.Tk()
window.title("SmartStopPlace")
window.geometry('1000x600')
path="E:/pictures/cp/1.jpg"
tk.Label(window,text="path:",font=('Arial', 12)).place(x=10,y=10)
e = tk.Entry(window,textvariable=path,width=30)
e.insert(0,path)
e.place(x=40,y=10)
tk.Button(window,text="获取图片",width=15,height=1,command=getImg).place(x=300,y=10)

frame_img=frame=tk.Frame(window)
frame.pack()

global photo
photo=ImageTk.PhotoImage(Image.open(path).resize((400,200)))
phote_label=tk.Label(window,width=400,height=200,image=photo)
phote_label.place(x=10,y=40)

tk.Label(window,text="车牌检测",font=('Arial', 12),background="green").place(x=10,y=250)
global photo2
photo2=ImageTk.PhotoImage(Image.open(path).resize((400,200)))
cp_label=tk.Label(window,width=400,height=200,image=photo)
cp_label.place(x=10,y=270)


tk.Label(window,text="二值文字识别图像",font=('Arial', 12),background="green").place(x=500,y=10)
global photo3

photo3=ImageTk.PhotoImage(Image.open(path).resize((400,200)))
binary_label_0=tk.Label(window,width=50,height=200,image=photo)
binary_label_0.place(x=500,y=40)
binary_label_1=tk.Label(window,width=50,height=200,image=photo)
binary_label_1.place(x=560,y=40)
binary_label_2=tk.Label(window,width=50,height=200,image=photo)
binary_label_2.place(x=620,y=40)
binary_label_3=tk.Label(window,width=50,height=200,image=photo)
binary_label_3.place(x=680,y=40)
binary_label_4=tk.Label(window,width=50,height=200,image=photo)
binary_label_4.place(x=740,y=40)
binary_label_5=tk.Label(window,width=50,height=200,image=photo)
binary_label_5.place(x=800,y=40)
binary_label_6=tk.Label(window,width=50,height=200,image=photo)
binary_label_6.place(x=860,y=40)


fram_all=tk.Frame(window)
fram_all.pack(side="bottom")
text_frame=tk.Frame(fram_all)
text_frame.pack()
cp_str="湘A00000"
tk.Label(text_frame,text="识别结果:",font=('Arial', 12),background="orange").pack(side="left")
str_Entry=tk.Entry(text_frame,textvariable=cp_str)
str_Entry.pack(side="left")
str_Entry.insert(0,cp_str)
frame=tk.Frame(fram_all,background="red",height=100)
frame.pack()
frame2=tk.Frame(fram_all,background="green",height=100)
frame2.pack()
tk.Label(frame,text="操作",font=('Arial', 12),background="#FFFF00",padx=10).pack(side="left")
inBtn=tk.Button(frame,text="进入停车场",padx=10,command=inAction)
inBtn.pack(side="left")
outBtn=tk.Button(frame,text="离开停车场",padx=10,command=outAction)
outBtn.pack(side="left")
searchBtn=tk.Button(frame,text="查询车牌信息",padx=10,command=getInfo)
searchBtn.pack(side="left")
OtherBtn=tk.Button(frame,text="other2",padx=10)
OtherBtn.pack(side="left")
inTime_label=tk.Label(frame2,text="进入时间",font=('Arial', 12),background="green",padx=50)
inTime_label.pack(side="left")
# tk.Label(frame2,text="    ",font=('Arial', 12),padx=10).pack(side="left")
inTime_Entry=tk.Entry(frame2,width=30)
inTime_Entry.pack(side="left")
outTime_label=tk.Label(frame2,text="离开时间",font=('Arial', 12),background="green",padx=50)
outTime_label.pack(side="left")
# tk.Label(frame2,text="    ",font=('Arial', 12),padx=10).pack(side="left")
outTime_Entry=tk.Entry(frame2,width=30)
outTime_Entry.pack(side="left")
total_Time_label=tk.Label(frame2,text="停车时长",font=('Arial',12),background="green",padx=50)
total_Time_label.pack(side="left")

window.mainloop()