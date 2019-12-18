# 项目简介

    Kaggle里包含了42000份训练数据和28000份测试数据（和谷歌准备的MNIST数据，在数量上有所不同），Kaggle的数据都是表格形式的，和MNIST给的图片不一样。但实际上只是对图片的信息进行了处理，把一个28*28的图片信息，变成了28*28=784的一行数据。

## [手写数字识别第一次尝试](https://github.com/yonglpy/data-analysis/edit/master/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/ANN.py)

第一次尝试采用的简单ANN模型：

1）使用一个最简单的单层的神经网络进行学习

2）用SoftMax来做为激活函数

3）用交叉熵来做损失函数

4）用梯度下降来做优化方式

训练后模型预测准确率为92%

## [手写数字识别第二次尝试](https://github.com/yonglpy/data-analysis/edit/master/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/Three_flood_CNN.py)

第二次尝试采用的三层CNN模型

1）激活函数选择的relu

2)用梯度优化方法选择的Aadelta

训练后模型预测准确率为98%

### [手写数字识别第三次尝试](https://github.com/yonglpy/data-analysis/edit/master/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/Multilayer_CNN_keras.py)

第三次尝试采用的模型是多层CNN(keras框架）

训练后模型预测准确率为99.38%
