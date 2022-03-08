import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


pd.options.display.max_columns = None
pd.options.display.max_rows = None
df=pd.read_csv(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\Filled_Data.csv',encoding='utf-8')
#df.iloc[:,11:].boxplot()
#plt.show()
#接下来对西装尺码进行空缺值填充
#print(df)

#对df进行抽样，并且提取其中完备的数据集下标
Sample=df.sample(n=300,replace=False,axis=0)
#
Sample=Sample[Sample['Suits'].notnull()]
#print(Sample)
X_Sample=pd.concat([Sample.loc[:,:'Hips'],Sample.loc[:,'Shoes']],axis=1)
#print(X_Sample)
Y_Sample=df.loc[:,'Suits'][X_Sample.index]
#print(Y_Sample)
#datasets_X是np数组
rest_unbid=[]
for x in df.index:
    if x not in X_Sample.index:
        rest_unbid.append(x)
#print(rest_unbid)
Data=df.loc[rest_unbid]
Train=Data[Data['Suits'].notnull()]
X_train=pd.concat([Train.iloc[:,:15],Train.iloc[:,16]],axis=1)
Y_train=Train.iloc[:,15]
predict_index=[]
for x in Data.index:
    if x not in Train.index:
        predict_index.append(x)
#print(predict_index)
Predict=Data.loc[predict_index]
#print(Train)
#print(Predict)
model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
X_Sample=X_Sample.values.tolist()
#print(X_Sample)
def P(x):
    sum=0
    for i in range(len(x)):
        sum=sum+b[i]*x[i]
    return sum+a

predict_result=[P(i) for i in X_Sample]
#print(predict_result)
#print(Y_Sample)
plt.plot(predict_result,label='predict_result')
plt.plot(Y_Sample.values.tolist(),label='Y_Sample')
plt.xlabel('The Under-bid of Data')
plt.ylabel('Size of Suits')
plt.title('The comparison between the predicted value and the real value')
plt.text(10,25,'About 400 pieces of data were extracted to test the results of Multiple linear regression filling missing values')

plt.legend()
plt.show()

Predict_X=pd.concat([Predict.iloc[:,:15],Predict.iloc[:,16]],axis=1)
Predict_X=Predict_X.values.tolist()
#print(Predict_X)
Predict.loc[:,'Suits']=[round(P(i)) for i in Predict_X]
#print(Predict)
#print(df)

df=pd.concat([Sample,Predict,Train],axis=0)
df=df.sort_index()
print(df)
df.to_csv(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\Filled_Data2.csv', mode='w', header=True,index=False)
