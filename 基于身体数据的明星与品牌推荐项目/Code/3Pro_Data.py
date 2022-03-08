#将所有的繁体字替换成简体字
#对眼睛颜色和皮肤颜色进行独热编码
#在发现有极个别的异常值之后决定使用统计学和机器学习的方法分辨异常值
#将洗干净的数据和异常值分别存放


from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import numpy as np
from pyod.models.knn import KNN

np.set_printoptions(threshold=np.inf)
f=open(r"C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\New_Data.txt",'r',encoding='utf-8')
f.readline()
line=f.readline()
data=[]
while line:
    line=line.strip().split(',')
    del line[-1]
    #print(line)
    data.append(line)
    #将繁体字替换成简体字
    if '淺啡' in line:
        line[line.index('淺啡')]='浅啡'
    if '綠色' in line:
        line[line.index('綠色')]='绿色'
    if '藍色' in line:
        line[line.index('藍色')]='蓝色'

    line=f.readline()

#对4,5,6列独热编码
ct = ColumnTransformer(
         [("enc", preprocessing.OneHotEncoder(), [4,5])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough')
data1=ct.fit_transform(data)
one=preprocessing.OneHotEncoder()
X=[[i[4],i[5]] for i in data]
#print(X)
Y=one.fit_transform(X)
print(one.categories_)
#print(type(data1))

Data=[]
for line in data1:
    line=list(line)
    for i in range(len(line)):
        try:
            line[i]=eval(line[i])
        except:
            pass
    Data.append(line)
#print(Data)

'''
#对异常值的检测与剔除
#3σ原则(拉依达准则)
def three_sigma(Ser1,p):
    Ser=Ser1.tolist()
    temp=[]
    for item in Ser1:
        if item=='':
            continue
        else:
            temp.append(eval(item))

    M=np.mean(temp)
    S=np.std(temp)
    outrange=[Ser.index(str(x)) for x in temp if x<M-p*S or x>M+p*S]
    return outrange

Data=np.array(Data)
#对11-15进行达意拉异常值检测
outrange=[]
for i in range(11,16):
    temp=Data[:,i]
    if i in (11,12,13,14):
        outrange.append(three_sigma(temp,4))
    else:
        outrange.append(three_sigma(temp, 5))
Data=Data.tolist()
for i in range(len(outrange)):
    print('\n\n')
    for x in outrange[i]:
        print(x,Data[x][11:17])
'''

#建立四个数据集，交叉验证异常值
X1=[]#空，空
X2=[]#空，不空
X3=[]#不空，空
X4=[]#不空，不空
for i in range(len(Data)):
    if Data[i][15]=='' and Data[i][16]=='':
        X1.append([i,*Data[i][11:15]])
    elif Data[i][15]=='':
        X2.append([i,*Data[i][11:15],Data[i][16]])
    elif Data[i][16] == '':
        X3.append([i,*Data[i][11:16]])
    else:
        X4.append([i,*Data[i][11:17]])

X=[]
X.append(X1)
X.append(X2)
X.append(X3)
X.append(X4)

#利用knn近邻方式检测异常值

# 训练一个kNN检测器
clf_name = 'kNN'
clf1 = KNN(contamination=0.05) # 初始化检测器clf
clf2 = KNN(contamination=0.05) # 初始化检测器clf
clf3 = KNN(contamination=0.05) # 初始化检测器clf
clf4 = KNN(contamination=0.05) # 初始化检测器clf
#先检测11到14列
clf1.fit_predict(X1) #使用X_train训练检测器clf
clf2.fit_predict(X2) #使用X_train训练检测器clf
clf3.fit_predict(X3) #使用X_train训练检测器clf
clf4.fit_predict(X4) #使用X_train训练检测器clf
# 返回训练数据X_train上的异常标签和异常分值
y=[]
y.append(clf1.labels_)  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
y.append(clf2.labels_)  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
y.append(clf3.labels_)  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
y.append(clf4.labels_)  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
outrange=[]

for i in range(4):
    for j in range(len(y[i])):
        if y[i][j]==1:
            #print(X[i][j])
            outrange.append(X[i][j][0])
f=open("Outrange_Data.txt",'w',encoding='utf-8')
#将异常值输出到文件中保存，并且将Data
for i in outrange:
    print(Data[i],file=f)
temp=Data[:]
for i in outrange:
    Data.remove(temp[i])
f=open("Pure_Data.txt",'w',encoding='utf-8')
for i in Data:
    print(i, file=f)
#print(help(KNN))









