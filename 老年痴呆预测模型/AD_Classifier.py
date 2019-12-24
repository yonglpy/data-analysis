# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:53:47 2019

@author: mayong
"""

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import random
from sklearn.externals import joblib

# 加载数据
feautre_data = pd.read_csv('D:/数据挖掘项目练习/AD/train_data.csv')
label_data = pd.read_csv('D:/数据挖掘项目练习/AD/1_preliminary_list_train.csv')

# 把结果标签设置0和1，其中1代表老年痴呆症，0代表正常
label_data = label_data.replace('AD',1)
label_data = label_data.replace('CTRL',0)

# 由于每个特征数值分布相差比较大，所以需要进行数据标准化处理
scaler = StandardScaler()
feautre_data_scaler = scaler.fit_transform(feautre_data.iloc[:,1:])
x_train_data =pd.DataFrame(data=feautre_data_scaler,columns=feautre_data.columns[1:])
col_name = list(x_train_data.columns)
col_name.insert(col_name.index(col_name[0]),'uuid')
x_train_data = x_train_data.reindex(columns=col_name)
x_train_data['uuid'] = feautre_data['uuid']

df_train_data = pd.merge(label_data,x_train_data,how='inner',on='uuid')

# 利用REFCV(递归特征消除方法)来筛选特征重要程度)
def FeaturesDeal(x_data,y_data,n):
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='accuracy')
    rfecv.fit(x_data, y_data)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Ranking of features : %s" % rfecv.ranking_)
    #plt.figure(figsize=(10,10))
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show()
    select_feature = []
    for i in range(len(rfecv.ranking_)):
        if rfecv.ranking_[i] <= n:
            select_feature.append(x_data.columns[i])
    return select_feature

feaure = df_train_data.iloc[:,2:]
target = df_train_data.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(feaure,target , test_size=0.2,random_state=1)

# 用支持向量机来训练模型
def SVMTrain(xa,ya,x,y,x_t,y_t,n):
    select_feature = FeaturesDeal(xa,ya,n)
    print('SVM use Features:{}'.format(select_feature))
    select_feature_data = x[select_feature]
    select_feature_data_t = x_t[select_feature]
    model = svm.SVC(C=1,kernel='linear',gamma=0.25)
    model.fit(select_feature_data,y)
    model.score(select_feature_data,y)
    predicted = model.predict(select_feature_data_t)
    #outdata = pd.DataFrame({'Readl':y_test_select,'Predicted':predicted})
    print('SVM prediction result')
    print ('查准率、查全率、F1值：')
    print (classification_report(y_t, predicted, target_names=None))
    return predicted,model
    #joblib.dump(model,'D:/数据挖掘项目练习/AD/AD_SVM.model')

# 用逻辑回归训练模型
def LR(xa,ya,x,y,x_t,y_t,n):
    select_feature = FeaturesDeal(xa,ya,n)
    print('LR use Feature:{}'.format(select_feature))
    select_feature_data = x[select_feature]
    select_feature_data_t = x_t[select_feature]
    LR_clf = linear_model.LogisticRegression(C=1,penalty='l1',tol=1e-4,solver='liblinear',max_iter=200)
    LR_clf.fit(select_feature_data,y)
    y_predict = LR_clf.predict(select_feature_data_t)
    print('LogisticRegression result')
    print('查准率、查全率、F1值：')
    print (classification_report(y_t, y_predict, target_names=None))
    return y_predict,LR_clf

# 用随机森林训练模型
def RF(xa,ya,x,y,x_t,y_t,n):
    select_feature = FeaturesDeal(xa,ya,n)
    print('RF use Feature:{}'.format(select_feature))
    select_feature_data = x[select_feature]
    select_feature_data_t = x_t[select_feature]
    #X_train, X_test, y_train, y_test = train_test_split(select_feature_data,y , test_size=0.2,random_state=1)
    rf = RandomForestClassifier(oob_score=True,random_state=10,n_estimators=70,max_depth=10)
    rf.fit(select_feature_data,y)
    predict = rf.predict(select_feature_data_t)
    print('RandomForest prediction result')
    print('查准率、查全率、F1值：')
    print (classification_report(y_t, predict, target_names=None))
    return predict,rf

# 用KNN算法训练模型
def KNN(xa,ya,x,y,x_t,y_t,n,m):
    select_feature = FeaturesDeal(xa,ya,n)
    print('KNN use Feature:{}'.format(select_feature))
    select_feature_data = x[select_feature]
    select_feature_data_t = x_t[select_feature]
    #X_train, X_test, y_train, y_test = train_test_split(select_feature_data,y , test_size=0.2,random_state=1)
    model = KNeighborsClassifier(n_neighbors=m)
    model.fit(select_feature_data,y)
    predict = model.predict(select_feature_data_t)
    print('KNeighborsClassifier prediction result')
    print('查准率、查全率、F1值：')
    print (classification_report(y_t, predict, target_names=None))
    return predict,model
# 用决策树算法训练模型
def DTC(xa,ya,x,y,x_t,y_t,n):
    select_feature = FeaturesDeal(xa,ya,n)
    print('DTC use Feature:{}'.format(select_feature))
    select_feature_data = x[select_feature]
    select_feature_data_t = x_t[select_feature]
    model = DecisionTreeClassifier()
    model.fit(select_feature_data,y)
    predicted = model.predict(select_feature_data_t)
    print('DecisionTreeClassifier prediction result')
    print('查准率、查全率、F1值：')
    print (classification_report(y_t, predicted, target_names=None))
    return predicted,model

DTC_predicted,DTC_model = DTC(feaure,target,X_train,y_train,X_test,y_test,18)
joblib.dump(DTC_model,'D:/数据挖掘项目练习/AD/AD_DTC.model')
SVM_predicted,SVM_model = SVMTrain(feaure,target,X_train,y_train,X_test,y_test,10)
joblib.dump(SVM_model,'D:/数据挖掘项目练习/AD/AD_SVM.model')
KNN_predicted,KNN_model = KNN(feaure,target,X_train,y_train,X_test,y_test,8,7)
joblib.dump(KNN_model,'D:/数据挖掘项目练习/AD/AD_KNN.model')
LR_predicted,LR_model = LR(feaure,target,X_train,y_train,X_test,y_test,13)
joblib.dump(LR_model,'D:/数据挖掘项目练习/AD/AD_LR.model')
RF_predicted,RF_model = RF(feaure,target,X_train,y_train,X_test,y_test,7)
joblib.dump(RF_model,'D:/数据挖掘项目练习/AD/AD_RF.model')
final_predicted = []
# 先用5个分类算法训练好的模型对验证数据集进行结果预测，然后取预测结果出现次数最多的，作为最后预测结果
for k in range(len(y_test)):
    a = []
    a.append(SVM_predicted[k])
    a.append(KNN_predicted[k])
    a.append(LR_predicted[k])
    a.append(RF_predicted[k])
    a.append(DTC_predicted[k])
    b = Counter(a)
    if b[1] > b[0]:
        final_predicted.append(1)
    if b[1] < b[0]:
        final_predicted.append(0)
    if b[1] == b[0]:
        final_predicted.append(random.sample([0,1],1)[0])
print('Final prediction result')
print('查准率、查全率、F1值：')
print (classification_report(y_test, final_predicted, target_names=None))