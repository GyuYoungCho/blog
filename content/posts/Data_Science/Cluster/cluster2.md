---
title: "K-means"
date: 2021-03-25T12:11:41+09:00
draft: false
categories : ["Data_Science", "Cluster"]
---

## K-means Clustering
` 각 군집에 할당된 포인트들의 평균 좌표를 이용해 중심점을 반복적으로 업데이트

1. 각 데이터에 대해 가까운 데이터 찾고 새로 할당된 군집 기반으로 새로운 중심 계산
2. 클러스터 할당이 바뀌지 않을 때까지 반복

- 거리는 Manhattan이나 Euclidean

#### K 설정 문제
최적화되 k를 찾기 어려움 -> Silhouette method 사용
	- 객체와 그 객체가 속한 군집의 데이터들과의 비 유사성을 계산
- a(i) : i와 i가 속한 군집 데이터들과의 비 유사성
- b(i) : i가 속하지 않은 다른 군집의 모든 데이터들과의 비 유사성의 최솟값(가장 가까운 군집)
- s(i) = $(b(i) - a(i)) / max{a(i), b(i)}
	- s(i)의 값이 1에 가까울수록 올바른 클러스터에 분류된 것이라 할 수 있음.
	- k를 증가시키며 평균 s(i), silhouette coefficient가 최대가 되는 k를 선택

<br>
K-means의 또 다른 문제는 이상치에 민감하다는 것인데 이를 해결하기 위해 K-medoids(중간점)를 사용
평균 대신 중앙값을 구한다고 생각하면 된다.

---

### apply
Iris 데이터를 활용하여 Kmeans clustering

```python
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
iris = datasets.load_iris()
```

```python
X = iris.data[:, :2]
y = iris.target
```

```python
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Spea1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
```

![image](https://user-images.githubusercontent.com/49333349/112422147-c8875580-8d73-11eb-87e9-bdb7e91b2e02.png)

```python
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(X)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=3, n_init=10, n_jobs=4, precompute_distances='auto',
        random_state=21, tol=0.0001, verbose=0)


각 군집의 center 보기

```python
centers = km.cluster_centers_
print(centers)
```

    [[5.77358491 2.69245283]
     [5.006      3.418     ]
     [6.81276596 3.07446809]]
    
실제 분류된 값과 군집으로 분류된 값 비교

```python
new_labels = km.labels_

fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
```

![image](https://user-images.githubusercontent.com/49333349/112422239-e654ba80-8d73-11eb-8cd3-72ab08631992.png)



#### 2차원의 가상 데이터에 Kmeans clustering


```python
from sklearn.datasets import make_blobs
# create dataset
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)
```


```python
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
y_km
```

    array([1, 0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 2, 0, 0, 2, 2, 1, 2, 1, 0, 1,
           0, 0, 2, 1, 1, 0, 2, 1, 2, 2, 2, 2, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1,
           1, 1, 2, 0, 2, 0, 1, 0, 0, 1, 1, 2, 0, 1, 2, 0, 2, 2, 2, 2, 0, 2,
           0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 1, 1, 2,
           2, 1, 1, 0, 1, 0, 1, 0, 2, 2, 1, 1, 1, 1, 2, 1, 1, 0, 2, 0, 0, 0,
           2, 0, 1, 2, 0, 2, 0, 0, 2, 2, 0, 1, 0, 0, 1, 1, 2, 1, 2, 2, 2, 2,
           1, 2, 2, 2, 0, 2, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1])

군집 분류 시각화

```python

plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
```

![image](https://user-images.githubusercontent.com/49333349/112422526-6418c600-8d74-11eb-8b97-07550991271c.png)



#### k 를 4로 할경우 


```python
km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
```


```python

plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)


plt.scatter(
    X[y_km == 3, 0], X[y_km == 3, 1],
    s=50, c='lightblue',
    marker='d', edgecolor='black',
    label='cluster 4'
)


# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
```

![image](https://user-images.githubusercontent.com/49333349/112422674-97f3eb80-8d74-11eb-8b5d-0ce54637d000.png)

한 군집이 2개로 나눠진 것처럼 보임. K=3 일 때가 더 명확하게 나온 것 같지만 K값을 변화시켜 최적의 K를 찾아보는 게 좋다.

#### 최적의 K 찾기

```python
from sklearn.metrics import silhouette_score
distortions = []
for i in range(2, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    
    distortions.append(silhouette_score(X,km.predict(X), metric='euclidean'))

plt.plot(range(2, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
```

![image](https://user-images.githubusercontent.com/49333349/112424036-0fc31580-8d77-11eb-8892-d2e8b3d50236.png)

간단하게는 kmeans 클래스의 inertia_를 이용해 최적의 k를 찾을 수 있다.

