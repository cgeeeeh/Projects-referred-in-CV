import pandas as pd
from numpy import nan as NaN
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
from matplotlib import pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None
def fill_missing_rf(X,Y):

    """
    使用随机森林填补一个特征的缺失值的函数

    参数：
    X:输入
    Y：label
    """
    # 找出训练集和测试集
    Ytrain = Y[Y.notnull()]#特征不缺失的值
    Ytest = Y[Y.isnull()]#特征缺失的值
    print(X.index)
    #print(Ytrain.index)
    #for x in Ytrain.index:
    #    try:
    #        a=X.loc[x]
    #    except:
    #        print(x)#特征不缺失的值对应其他n-1个特征+本来的标签
    Xtrain = X.loc[Ytrain.index]
    Xtest = X.loc[Ytest.index]#特征缺失的值对应其他n-1个特征+本来的标签
#
    #用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    #print(np.isnan(Xtest).any())
    Ypredict = rfr.predict(Xtest)
    return rfr,Ytest.index,Ypredict

f=open(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\Pure_Data.txt','r',encoding='utf-8')
context=[]

line=f.readline()
while line:
    line=line.strip().strip('[').strip(']').split(',')
    for i in range(len(line)):
        line[i]=eval(line[i])
        if line[i]=='':
            line[i]=NaN
    context.append(line)
    line = f.readline()
#print(context)
df=pd.DataFrame(context,dtype=float,columns=['Skin', 'Skin', 'Skin',
                                             'Eyes','Eyes','Eyes','Eyes','Eyes','Eyes','Eyes','Eyes',
                                             'Height','Chest','Waist','Hips',
                                             'Suits','Shoes'])
#对数组进行随机抽样，然后检测随机森林填补缺失值的准确度
y=df.iloc[:,-1]
y_Sample = y.sample(frac=0.1,replace=False,random_state=None,axis=0)
y_Sample=y_Sample[y_Sample.notnull()]
#print(y_Sample)
X=df.iloc[:,:15]
X_Sample=X.iloc[y_Sample.index]
#print(pd.concat([X_Sample,y_Sample],axis=1))
#print(y_Sample.index)
#print(X.index[-1])
Rest_index=[]
for x in X.index:
    if x not in y_Sample.index:
        Rest_index.append(x)
#print(Rest_index)
Rest_index=pd.Int64Index(Rest_index)
#print(Rest_index)
#print(df)
X=df.iloc[Rest_index,:15]
#print(X)
y=df.iloc[Rest_index,-1]
#print(y)
rfr,y_index,y_pre=fill_missing_rf(X,y)

#对之前选取出来的样本值进行比较验证
Y_Sample_predict = rfr.predict(X_Sample)
#print((Y_Sample_predict-y_Sample).values)
#对比较结果进行可视化展示
plt.plot(Y_Sample_predict,label='predict value')
plt.plot(y_Sample.values,label='real value')
plt.ylabel('Difference between predicted Shoes Size and real Shoes Size')
plt.xlabel('The Under-bid of Data')
plt.legend(['predict value','real value'])
plt.title('The comparison between the predicted value and the real value')
plt.text(-20,31,'About 400 pieces of data were extracted to test the results of random forest fill missing values')
plt.show()
y_pre=[round(x) for x in y_pre]
for i in range(len(y_index)):
    df.iloc[y_index[i], -1]=y_pre[i]

df.to_csv(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\Filled_Data.csv', mode='w', header=True,index=False)






