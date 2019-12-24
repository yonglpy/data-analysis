# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:44:38 2019

@author: mayong
"""

import pandas as pd
from sklearn.externals import joblib
from collections import Counter

prediction_data = pd.read_csv('D:/数据挖掘项目练习/AD/test_data_scaler.csv')
# 每个算法使用特征数量是有差异的，下面这些数组记录每个算法使用特征，然后根据这些特征再结合训练好的模型进行预测
DTC_feature_list = ['F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 
                    'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm',
                    'shimmerLocaldB_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 
                    'hammarbergIndexV_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_stddevNorm',
                    'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 
                    'hammarbergIndexUV_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'MeanUnvoicedSegmentLength', 
                    'StddevUnvoicedSegmentLength']
SVM_feature_list = ['F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 
                    'slopeV0-500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_stddevNorm', 
                    'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 
                    'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength']
KNN_feature_list = ['slopeV0-500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_stddevNorm', 
                    'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean',
                    'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength']

LR_feature_list = ['F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 
                   'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'shimmerLocaldB_sma3nz_stddevNorm', 
                   'hammarbergIndexV_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_stddevNorm',
                   'mfcc1V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean',
                   'slopeUV500-1500_sma3nz_amean', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength']
RF_feature_list = ['slopeV0-500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean',
                   'hammarbergIndexUV_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'MeanUnvoicedSegmentLength', 
                   'StddevUnvoicedSegmentLength']
# 根据特征提前使用数据集
DTC_data = prediction_data[DTC_feature_list]
SVM_data = prediction_data[SVM_feature_list]
KNN_data = prediction_data[KNN_feature_list]
LR_data = prediction_data[LR_feature_list]
RF_data = prediction_data[RF_feature_list]

# 加载已经训练好的模型
DTC_model = joblib.load('D:/数据挖掘项目练习/AD/AD_DTC.model')
SVM_model = joblib.load('D:/数据挖掘项目练习/AD/AD_SVM.model')
KNN_model = joblib.load('D:/数据挖掘项目练习/AD/AD_KNN.model')
LR_model = joblib.load('D:/数据挖掘项目练习/AD/AD_LR.model')
RF_model = joblib.load('D:/数据挖掘项目练习/AD/AD_RF.model')

#预测结果存放
DTC_predicted = DTC_model.predict(DTC_data)
SVM_predicted = SVM_model.predict(SVM_data)
KNN_predicted = KNN_model.predict(KNN_data)
LR_predicted = LR_model.predict(LR_data)
RF_predicted = RF_model.predict(RF_data)

# 结果采用投票方法确定最终预测结果
final_predicted = []
for j in range(len(DTC_predicted)):
    a = []
    a.append(DTC_predicted[j])
    a.append(SVM_predicted[j])
    a.append(KNN_predicted[j])
    a.append(LR_predicted[j])
    a.append(RF_predicted[j])
    b = Counter(a)
    if b[1] > b[0]:
        final_predicted.append(1)
    if b[1] < b[0]:
        final_predicted.append(0)
# 输出最终预测结果
print(final_predicted)