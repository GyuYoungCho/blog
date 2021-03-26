---
title: "SVM"
date: 2021-03-26T01:15:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---

## Support Vector Machine

`가장 가까운 K개 점을 선택헤 분류 및 예측

```python
import numpy as np
import matplotlib.pyplot as plt
```

- 함수 불러오기


```python
from sklearn import svm, datasets
iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target

C=1
clf=svm.SVC(kernel='linear',C=C)
clf.fit(X,y)
```




    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
from sklearn.metrics import confusion_matrix
y_pred=clf.predict(X)
confusion_matrix(y,y_pred)  
```




    array([[50,  0,  0],
           [ 0, 38, 12],
           [ 0, 15, 35]])



# 2. kernel SVM 적합 및 비교

- LinearSVC


```python
clf=svm.LinearSVC(C=C,max_iter=10000)
clf.fit(X,y)
y_pred=clf.predict(X)
confusion_matrix(y,y_pred)
```




    array([[49,  1,  0],
           [ 2, 30, 18],
           [ 0,  9, 41]])



- radial basis function


```python
clf=svm.SVC(kernel='rbf',gamma=0.7,C=C,max_iter=10000)
clf.fit(X,y)
y_pred=clf.predict(X)
confusion_matrix(y,y_pred)
```




    array([[50,  0,  0],
           [ 0, 37, 13],
           [ 0, 13, 37]])



- polynomial kernel


```python
clf=svm.SVC(kernel='poly',degree=3,C=C,gamma='auto')
clf.fit(X,y)
y_pred=clf.predict(X)
confusion_matrix(y,y_pred)
```




    array([[50,  0,  0],
           [ 0, 38, 12],
           [ 0, 16, 34]])



- 시각적 비교

- 함수 정의


```python
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
```

- 데이터 불러오기


```python
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target
```

- 모델정의 및 피팅


```python
C = 1.0 #Regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)
```


```python
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')
```


```python
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
```


![image](https://user-images.githubusercontent.com/49333349/112592902-13739c80-8e4a-11eb-8000-9a733877095b.png)
