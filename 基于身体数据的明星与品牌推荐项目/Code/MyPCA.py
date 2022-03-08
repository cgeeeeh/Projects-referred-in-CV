import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets
from sklearn.manifold import TSNE


pd.options.display.max_columns = None
pd.options.display.max_rows = None
X=pd.read_csv(r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\Filled_Data2.csv',header=1)
def PCA_Show():
    # 降维
    #0.95 4
    #0.868 2
    pca = PCA(n_components=0.868)
    pca.fit(X)
    # 输出特征值
    #print(pca.explained_variance_)
    #输出特征向量
    #print(pca.components_)
    # 降维后的数据
    X_new = pca.transform(X)
    #print(X_new.shape,X_new)
    print(help(pca))
    print(pca.components_)
    print(pca.explained_variance_)
    fig = plt.figure()
    plt.title('The result of PCA')
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='.')
    plt.xlabel('The 1st dimension of Data')
    plt.ylabel('The 2nd dimension of Data')
    plt.text(-30,-45,'The data was reduced to two dimensions, covering 86% of the original data')
    plt.show()

def MyPCA():
    pca = PCA(n_components=0.95)
    pca.fit(X)
    # 输出特征值
    #print(pca.explained_variance_)
    #输出特征向量
    #print(pca.components_)
    # 降维后的数据
    X_new = pca.transform(X)
    #print(X_new.shape,X_new)
    #print(help(pca))
    print("pca降维时各个特征的权重是",pca.explained_variance_ratio_)
    #print(pca.explained_variance_)
    return pca,X_new,pca.explained_variance_ratio_





#PCA_Show()



