---
title: "Clustering 소개, Hierarchical clustering"
date: 2021-03-25T01:41:41+09:00
draft: false
categories : ["Data_Science", "Cluster"]
---

## 군집 분석
` 각 데이터의 유사성을 측정하여 높은 대상 집단을 분류하고 군집 간에 상이성을 규명

- K-means : 사용자가 지정한 k개의 군집으로 나누기
- Hierarchical : 나무 모양의 계층 구조를 형성해 나감.
- DBSCAN : 밀도 기반 군집, K개 설정 필요없음.

## Hierarchical Clustering
- 가까운 집단부터 계층적으로 묶어나감
- dendogram을 통해 시각화 가능
- 군집의 개수를 정하지 않아도 되나 데이터가 많을 경우 시각화나 많은 계층으로 나누기가 힘들어 데이터가 적으면 보기 좋음.

#### 방법
1. 모든 개체들 사이의 유사도 행렬 계산
2. 거리가 인접한 관측치끼리 cluster 형성
3. 유사도 행렬 update

아래와 같이 A와D가 같은 군집으로 묶이면 AD로 묶임
![image](https://user-images.githubusercontent.com/49333349/112417054-2adb5880-8d6a-11eb-8414-7c0e3dff16a4.png)


- 유사도 계산은 cluster 거리의 최소, 최대를 기준으로 하거나 평균, centroid 기준으로 계산됨. 
- 주로  ward 연결법이 쓰인다.

### apply

```python
customer_data = pd.read_csv('./data/shopping-data.csv')
customer_data.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Genre</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



```python
data = customer_data.iloc[:, 3:5].values
```
**덴드로그램 보기**

```python
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
# ward가 기준
dend = shc.dendrogram(shc.linkage(data, method='ward'))
```

![image](https://user-images.githubusercontent.com/49333349/112416659-6d506580-8d69-11eb-8dcb-dc2adb80fe88.png)


#### 군집 시각화

```python
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
```

    array([4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
           4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1,
           4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 0, 2, 0, 2,
           1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2,
           0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
           0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
           0, 2], dtype=int64)


```python
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
```


![image](https://user-images.githubusercontent.com/49333349/112416601-56117800-8d69-11eb-9c5d-3686c91bc52e.png)
