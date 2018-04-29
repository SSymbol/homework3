
#coding=utf-8
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans,  MeanShift, MiniBatchKMeans
from sklearn.metrics import classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('e:/hw3/data/train.csv')
result = open(r'e:/hw3/result/cluter_result.txt','a+')

#预处理
data.drop(['Name'], 1, inplace=True)
data.convert_objects(convert_numeric=True)
data.fillna(0, inplace = True )
label = data['Survived']
data.drop(['Survived'], 1, inplace=True)

#对类别型特征进行转化,成为特征向量
vec=DictVectorizer(sparse=False)
_data=vec.fit_transform(data.to_dict(orient='record'))
_data = preprocessing.scale(_data)

#使用KMean进行聚类模型的训练以及预测分析
clu_kmeans = KMeans(n_clusters=2)
kmeans_pred = clu_kmeans.fit_predict(_data)
print('KMeans:\n',classification_report(kmeans_pred,label))
print('KMeans:\n',classification_report(kmeans_pred,label),file=result)

#使用MeanShift进行聚类模型的训练以及预测分析
clu_mean = MeanShift()
mean_pred = clu_mean.fit_predict(_data)
print('MeanShift:\n',classification_report(mean_pred,label))
print('MeanShift:\n',classification_report(mean_pred,label),file=result)


#绘图
color_value = lambda a: 'r' if a == 1 else 'b'
#原始图像
color_ori = [color_value(d) for d in label]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['Age'],data['Pclass'],data['Fare'],c = color_ori,marker = 'o')
ax.set_xlabel('Age')
ax.set_ylabel('Pclass')
ax.set_zlabel('Fare')
plt.savefig('e:/hw3/output/cluster_original.png')
plt.clf()

#KMeans结果
color_kmeans = [color_value(d) for d in kmeans_pred]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['Age'],data['Pclass'],data['Fare'],c = color_kmeans,marker = 'o')
ax.set_xlabel('Age')
ax.set_ylabel('Pclass')
ax.set_zlabel('Fare')
plt.savefig('e:/hw3/output/cluster_result_kmeans.png')
plt.clf()

#meanshift结果
color_meanshift = [color_value(d) for d in mean_pred]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['Age'],data['Pclass'],data['Fare'],c = color_meanshift,marker = 'o')
ax.set_xlabel('Age')
ax.set_ylabel('Pclass')
ax.set_zlabel('Fare')
plt.savefig('e:/hw3/output/cluster_result_meanshift.png')
plt.clf()


