---
title: "KNN"
date: 2021-03-26T00:15:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---

## KNN

`가장 가까운 K개 점을 선택헤 분류 및 예측

#### iris datasets으로 KNN 적용

1. 모델 예측 및 confusion matrix 보기

```python
from sklearn import neighbors, datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
```


```python
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target
```

- 모델 구축


```python
clf = neighbors.KNeighborsClassifier(5)
clf.fit(X,y)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=5, p=2,
               weights='uniform')




```python
y_pred=clf.predict(X)
```


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)
```




    array([[49,  1,  0],
           [ 0, 38, 12],
           [ 0, 12, 38]], dtype=int64)



### 2.Cross-validation을 활용한 최적의 k찾기

```python
from sklearn.model_selection import cross_val_score
```

- CV 진행


```python
k_range = range(1,100)
k_scores= []

for k in k_range:
    knn=neighbors.KNeighborsClassifier(k)
    scores=cross_val_score(knn,X,y,cv=10,scoring="accuracy")
    k_scores.append(scores.mean())
```


```python
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()
```

![image](https://user-images.githubusercontent.com/49333349/112577287-6db73200-8e37-11eb-96fb-95ab2ec05aff.png)


#### 가중치를 준 KNN


```python
n_neighbors = 40

h = .02  # step size in the mesh

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
```


![image](https://user-images.githubusercontent.com/49333349/112577593-f8982c80-8e37-11eb-9e29-e7d0dc378315.png)

![image](https://user-images.githubusercontent.com/49333349/112577618-051c8500-8e38-11eb-8984-eb223df3f1f1.png)


