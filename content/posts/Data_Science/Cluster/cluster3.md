---
title: "DBSCAN"
date: 2021-03-25T14:00:41+09:00
draft: false
categories : ["Data_Science", "Cluster"]
---

## DBSCAN
` 밀도 기반 군집
- 초기값에 민감하고 이상치 민감한 K-means 문제 해결

#### 방법
- eps-neighbors : epsilon 거리 이내의 데이터들을 한 군집으로 구성
- minPts : minPts보다 같거나 많은 데이터로 구성, minPts보다 적은 수의 데이터가 eps를 형성하면 noise로 취급 -> -1

#### hyper parameter 정하기
**minPts**
- 보통 3 이상으로 설정

**eps**
- 값을 변경하며 찾아도 되지만 KNN의 distance를 그래프로 나타내고 거리가 급격하게 증가하는 지점을 eps로 설정하는 게 좋음.

---

### apply


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
```


```python
data = np.load('./data/clusterable_data.npy')
```


```python
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
```

![image](https://user-images.githubusercontent.com/49333349/112428185-1dc86480-8d7e-11eb-870b-fc89cf749330.png)


```python
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
```


```python
plot_clusters(data, cluster.KMeans, (), {'n_clusters':3})
```


![image](https://user-images.githubusercontent.com/49333349/112428132-0ab59480-8d7e-11eb-94bd-24f73d5725fe.png)



```python
plot_clusters(data, cluster.KMeans, (), {'n_clusters':4})
```


![image](https://user-images.githubusercontent.com/49333349/112428111-012c2c80-8d7e-11eb-8f9e-9b7ec1e7d374.png)



```python
plot_clusters(data, cluster.KMeans, (), {'n_clusters':5})
```

![image](https://user-images.githubusercontent.com/49333349/112428085-f4a7d400-8d7d-11eb-930d-46ecee4d00ca.png)


```python
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.020})
```


![image](https://user-images.githubusercontent.com/49333349/112427958-d0e48e00-8d7d-11eb-916b-fc0601173ec9.png)



```python
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.03})
```

![image](https://user-images.githubusercontent.com/49333349/112427864-a266b300-8d7d-11eb-97c5-53628c1d1f34.png)


```python
dbs = DBSCAN(eps=0.03)
dbs2=dbs.fit(data)

```


```python
dbs2.labels_
```


    array([ 0,  0,  0, ..., -1, -1,  0], dtype=int64)



### HDBSCAN
DBSCAN의 발전, hyper parameter에 덜 민감하다.


```python
import hdbscan
```


```python
plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':45})
```

![image](https://user-images.githubusercontent.com/49333349/112426205-df7d7600-8d7a-11eb-9b96-43d73ba69c0f.png)


