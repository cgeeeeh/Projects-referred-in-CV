from MyPCA import MyPCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
np.set_printoptions(threshold=np.inf)
pd.options.display.max_columns = None
pd.options.display.max_rows = None



X_new,components=MyPCA()
X_new=pd.DataFrame(X_new)
model = KMeans(n_clusters = 10) #分为k类
model.fit(X_new)

r1=pd.Series(model.labels_).value_counts()
r2=pd.DataFrame(model.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
#print(r)
r.to_csv(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\KMeans_Result.csv',header=0,index=0)
#详细输出原始数据及其类别
r = pd.concat([X_new, pd.Series(model.labels_, index = X_new.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(X_new.columns) + [u'聚类类别'] #重命名表头
r.to_csv(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\K_outputfile.csv',index=0,encoding='utf-8') #保存结果

#用TSNE进行数据降维并展示聚类结果

tsne = TSNE()
tsne.fit_transform(X_new) #进行数据降维,并返回结果
tsne = pd.DataFrame(tsne.embedding_, index = X_new.index) #转换数据格式
print(tsne)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#不同类别用不同颜色和样式绘图
d = tsne[r[u'聚类类别'] == 0]     #找出聚类类别为0的数据对应的降维结果
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 3]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 4]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 5]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 6]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 7]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 8]
plt.plot(d[0], d[1], '.')
d = tsne[r[u'聚类类别'] == 9]
plt.plot(d[0], d[1], '.')
plt.title('The result of K_means whose dimension is reduced by T-SNE')
plt.xlabel('1st dimension of T-SNE')
plt.xlabel('2nd dimension of T-SNE')
plt.text(-90,-130,'The result of PCA is put into a K-Means machine(cluster=10), the result of which is shown after being processed by T-SNE')
plt.show()

