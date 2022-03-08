#!/usr/bin/env python
# -*- coding:utf-8 -*-
from MyPCA import MyPCA
def tran_Eye(C):#['白色', '黄色', '黑色']
    if C=='白色':
        return [1,0,0]
    if C=='黄色':
        return [0,1,0]
    if C=='黑色':
        return [0,0,1]
def tran_hair(C):#['棕色', '浅啡', '深啡', '灰色', '绿色', '蓝色', '褐色', '黑色']
    if C=='棕色':
        return [1,0,0,0,0,0,0,0]
    if C=='浅啡':
        return [0,1,0,0,0,0,0,0]
    if C=='深啡':
        return [0,0,1,0,0,0,0,0]
    if C=='灰色':
        return [0,0,0,1,0,0,0,0]
    if C=='绿色':
        return [0,0,0,0,1,0,0,0]
    if C=='蓝色':
        return [0,0,0,0,0,1,0,0]
    if C=='褐色':
        return [0,0,0,0,0,0,1,0]
    if C=='黑色':
        return [0,0,0,0,0,0,0,1]

import tkinter
win = tkinter.Tk()
win.title("Brand Recommend Made by CGH")
win.geometry("400x400+200+50")
f_1 = tkinter.Frame(win)
f_1.place(x=100,y=0)
l_1 = tkinter.Label(f_1,text="Skin")
l_1.pack()
l_2 = tkinter.Label(f_1,text="Eyes")
l_2.pack()
l_3 = tkinter.Label(f_1,text="Height")
l_3.pack()
l_4 = tkinter.Label(f_1,text="Chest")
l_4.pack()
l_5 = tkinter.Label(f_1,text="Waist")
l_5.pack()
l_6 = tkinter.Label(f_1,text="Hips")
l_6.pack()
l_7 = tkinter.Label(f_1,text="Suit")
l_7.pack()
l_8 = tkinter.Label(f_1,text="Shoes")
l_8.pack()


f_2 = tkinter.Frame(win)
f_2.place(x=170,y=0)
e1 = tkinter.Variable()
e2=tkinter.Variable()
e3=tkinter.Variable()
e4=tkinter.Variable()
e5=tkinter.Variable()
e6=tkinter.Variable()
e7=tkinter.Variable()
e8=tkinter.Variable()
entry1=tkinter.Entry(f_2,textvariable=e1)
entry2=tkinter.Entry(f_2,textvariable=e2)
entry3=tkinter.Entry(f_2,textvariable=e3)
entry4=tkinter.Entry(f_2,textvariable=e4)
entry5=tkinter.Entry(f_2,textvariable=e5)
entry6=tkinter.Entry(f_2,textvariable=e6)
entry7=tkinter.Entry(f_2,textvariable=e7)
entry8=tkinter.Entry(f_2,textvariable=e8)
entry1.pack()
entry2.pack()
entry3.pack()
entry4.pack()
entry5.pack()
entry6.pack()
entry7.pack()
entry8.pack()


# e就代表输入框这个对象
# 设置值
#e.set("wewewewewewe")
## 取值
#print(e.get())
#print(entry2.get())
import csv
import tkinter.messagebox as tkMessageBox
def get_info():
    print(e1.get())
    input=[]
    input.append(e1.get())
    input.append(e2.get())
    input.append(eval(e3.get()))
    input.append(eval(e4.get()))
    input.append(eval(e5.get()))
    input.append(eval(e6.get()))
    input.append(eval(e7.get()))
    input.append(eval(e8.get()))
    replace0=tran_Eye(input[0])
    replace1=tran_hair(input[1])
    input=replace0+replace1+input[2:]
    print(input)
    pca,X_new, components = MyPCA()
    PCA_result=pca.transform([input])[0]
    print(PCA_result)
    center=[]
    with open(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\KMeans_Result.csv', 'r') as f:        # 采用b的方式处理可以省去很多问题
        reader = csv.reader(f)
        for row in reader:
            center.append([eval(i) for i in row[:-1]])
    print(center)
    final=0
    compared=1000000000000
    for c in range(len(center)):
        s=0
        for i in range(4):
            s=s+(center[c][i]-PCA_result[i])*(center[c][i]-PCA_result[i])
        if s < compared:
            final=c
    print(c)
    #吴彦祖 黄色 黑色 183 91.44  55.88 81.28 43 41
    if c==9:
        star='吴彦祖'
        brand='利郎服饰,DESCENTE,欧莱雅'
        tkMessageBox.showinfo(star, brand)
B = tkinter.Button(win,text="推荐明星与品牌",command=lambda:get_info())

B.place(x=170,y=200)
win.mainloop()

