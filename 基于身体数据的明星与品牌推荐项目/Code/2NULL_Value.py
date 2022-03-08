#对缺失值的处理
#如果存在身高，三围，三色（肤色，眼睛颜色，头发发色）为空值，则抛弃这条目
#将所有的‘-’与‘0’均设置成空值
#去除掉所有单位之中的单位值
import csv

Suits_=[]
Shoes_=[]
Suits_P=[]
Shoes_P=[]
file=open('MODEL.csv','r',encoding='GBK')
reader=csv.reader(file)
new_data=[]
#'Height'0, 'Chest'1, 'Waist'2, 'Hips'3, 'Skin'4, 'Eyes'5, 'Hair'6, 'Suit'7, 'Shoes'8
for raw in reader:
    #print(raw)
    newline=raw[7:16]
    #如果存在身高，三围，三色（肤色，眼睛颜色，头发发色）为空值，则抛弃这条目
    error_set=(*newline[0:4],*newline[6:9])
    #print(error_set)
    if '' in error_set or '0' in error_set or '-' in error_set or '-1' in error_set:
        continue
    del newline[4]
    del newline[4]
    newline.append(raw[11])
    newline.append(raw[12])
    #将所有的‘-’与‘0’均设置成空值
    if newline[-1]=='0' or newline[-1]=='-' or newline[-1]=='00':
        newline[-1]=''
    if newline[-2]=='0' or newline[-2]=='-' or newline[-2]=='00':
        newline[-2]=''
    if 'cm' in newline[-1] or '码' in newline[-1]:
        newline[-1]=newline[-1].strip('cm')[0].strip('码')[0]
    if 'cm' in newline[-2] or '码' in newline[-2]:
        newline[-2]=newline[-2].strip('cm')[0].strip('码')[0]
    new_data.append(newline)
file.close()

file=open('MODEL1.csv','r',encoding='utf-8')
reader=csv.reader(file)

flag=1
for raw in reader:
    if flag:
        flag=0
        continue
    newline=raw[7:16]
    #如果存在身高，三围，三色（肤色，眼睛颜色，头发发色）为空值，则抛弃这条目
    error_set=(*newline[0:4],*newline[6:9])
    #print(error_set)
    if '' in error_set or '0' in error_set or '-' in error_set or '-1' in error_set:
        continue
    del newline[4]
    del newline[4]
    newline.append(raw[11])
    newline.append(raw[12])
    #将所有的‘-’与‘0’均设置成空值
    if newline[-1]=='0' or newline[-1]=='-' or newline[-1]=='00':
        newline[-1]=''
    if newline[-2]=='0' or newline[-2]=='-' or newline[-2]=='00':
        newline[-2]=''
    if 'cm' in newline[-1] or '码' in newline[-1]:
        newline[-1]=newline[-1].strip('cm')[0].strip('码')[0]
    if 'cm' in newline[-2] or '码' in newline[-2]:
        newline[-2]=newline[-2].strip('cm')[0].strip('码')[0]
    new_data.append(newline)
file.close()

#将数据写入txt文件存储
f=open("New_Data.txt",'w',encoding='utf-8')
for raw in new_data:
    print(raw)
    for i in range(len(raw)):
        if i!=6:
            print(raw[i],file=f,end=',')
    print("",file=f)

#统计已知Suits（Shoes ）的条目数量
count_Suits=0
count_Shoes=0
for raw in new_data:
    if raw[-2]!='':
        count_Suits=count_Suits+1
    if raw[-1]!='':
        count_Shoes=count_Shoes+1
print(count_Suits,count_Shoes)






