import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

def get_regression(x,y):


    # 将 x，y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    poly = PolynomialFeatures(degree = 5)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    return poly,lin2

    # 画图
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # x = np.arange(1,1000)
    #
    # x = x[:, np.newaxis]
    #
    # # y = np.sort(np.random.normal(5, 2, 9 ))
    # plt.scatter(x, y, label='实际值') # 散点图
    # # plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
    # plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red', label='预测值')
    # plt.legend() # 显示图例，即每条线对应 label 中的内容
    # plt.show() # 显示图形
    #
