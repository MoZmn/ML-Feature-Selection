# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing libraries
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import sklearn.decomposition
import numpy as np
import pandas as pd
#checking dataset

bcd = load_boston()
data = pd.DataFrame(bcd.data, columns = bcd.feature_names)
data['target'] = bcd.target
x = bcd.data[:, np.arange(0, 10)]
y  = bcd.target

n = x.shape[0]
print("x: ", x.shape)
print("n: ", n)
print()

m = y.shape[0]
print("y: ", y.shape)
print("m: ", m)
print()

data = pd.DataFrame(x, columns = np.arange(1, 11))
data['target'] = y

print('---------------------cov&mean matrix---------------')
print()
list1= list(x[:,0])
list2= list(x[:,1])
list3= list(x[:,2])
a = np.array([list1, list2, list3])
cov_mat = np.cov(a)
mean_mat = np.mean(x, axis=0)
print('cov matrix: ',cov_mat)
print()
print('mean matrix: ',mean_mat)
print()

print('---------------------PCA-----------------------------')
# linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#evaluation pf pca
from sklearn.decomposition import PCA
error_variances = []
mse = []
for i in range(10, 4, -1):
    pca = PCA(n_components = i)
    x = pca.fit_transform(x)
    model.fit(x, y)
    y_pred = model.predict(x)
    mse.append(mean_squared_error(y, y_pred))
    a = pca.explained_variance_ratio_
    summ = sum(a)
    error_variances.append(1-summ)
    
    n = x.shape[0]
    print("x: ", x.shape)
    print("n: ", n)
    print()

    mu = np.mean(x, axis=0)

    pca = sklearn.decomposition.PCA()
    pca.fit(x)

    nComp = i
    Xhat = np.dot(pca.transform(x)[:,:nComp], pca.components_[:nComp,:])
    Xhat += mu
    
    z = Xhat.shape[0]
    print("Xhat: ", Xhat.shape)
    print("z: ", z)
    print()


#creating plots
import matplotlib.pyplot as plt
plt.plot(np.arange(10, 4, -1), error_variances)
plt.title('classification error')
plt.show()
plt.plot(np.arange(10, 4, -1), mse, color = 'red')
plt.title('MSE')
plt.show()

print('-----------------------------FLD--------------------')
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
bcd = load_breast_cancer()
data = pd.DataFrame(bcd.data, columns = bcd.feature_names)
data['target'] = bcd.target
x = bcd.data
y = bcd.target
model = LinearDiscriminantAnalysis()
error_variances = []
for i in range(10, 4, -1):
    pca = PCA(n_components = i)
    x = pca.fit_transform(x)
    model.fit(x, y)
    y_pred = model.predict(x)
    mse.append(mean_squared_error(y, y_pred))
    a = pca.explained_variance_ratio_
    summ = sum(a)
    error_variances.append(1-summ)
    
    n = x.shape[0]
    print("x: ", x.shape)
    print("n: ", n)
    print()

import matplotlib.pyplot as plt
plt.plot(np.arange(10, 4, -1), error_variances)
plt.title('classification error')
plt.show()


