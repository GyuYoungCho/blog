---
title: "PCA"
date: 2021-03-27T00:15:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---

```python
from sklearn import datasets
from sklearn.decomposition import PCA
```

` PCA 함수를 활용하여 PC를 얻어냄. 아래의 경우 PC 2개를 뽑아냄.


```python
pca=PCA(n_components=2)
pca.fit(X)
```




    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)



- 아래와 같이 PC score를 얻어냄. 아래의 PC score를 이용하여, 회귀분석에 활용할 수 있음.


```python
PCscore=pca.transform(X)
PCscore[0:5]
```




    array([[-2.4608061 , -0.24553253],
           [-2.53956302, -0.06169198],
           [-2.71024021,  0.08277011],
           [-2.56577812,  0.2534473 ],
           [-2.50018456, -0.15361226]])




```python
eigens_v=pca.components_.transpose()
print(eigens_v)
```

    [[ 0.39378459 -0.91920275]
     [ 0.91920275  0.39378459]]
    


```python
mX=np.matrix(X)
for i in range(X.shape[1]):
    mX[:,i]=mX[:,i]-np.mean(X[:,i])
dfmX=pd.DataFrame(mX)
```


```python
(mX*eigens_v)[0:5]
```




    matrix([[-2.4608061 , -0.24553253],
            [-2.53956302, -0.06169198],
            [-2.71024021,  0.08277011],
            [-2.56577812,  0.2534473 ],
            [-2.50018456, -0.15361226]])



```python
plt.scatter(dfmX[0],dfmX[1])
origin = [0], [0] # origin point
plt.quiver(*origin, eigens_v[0,:], eigens_v[1,:], color=['r','b'], scale=3)
plt.show()
```

![image](https://user-images.githubusercontent.com/49333349/112685102-69325e00-8eb7-11eb-8cf0-999f2c069c38.png)



### PC를 활용한 회귀분석

- 이번에는 모든 독립변수를 활용하여 PC를 뽑아냄.


```python
X2 = iris.data
pca2 = PCA(n_components=4)
pca2.fit(X2)
```




    PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
pca2.explained_variance_
```




    array([ 4.19667516,  0.24062861,  0.07800042,  0.02352514])




```python
PCs=pca2.transform(X2)[:,0:2]
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
```

- 모델의 복잡성으로 인하여 기존 자료를 이용한 분석은 수렴하지 않는 모습.


```python
clf = LogisticRegression(solver="sag",multi_class="multinomial").fit(X2,y)
```

    d:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\sag.py:286: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      "the coef_ did not converge", ConvergenceWarning)
    

- PC 2개 만을 뽑아내여 분석한 경우 모델이 수렴.


```python
clf2 = LogisticRegression(solver="sag",multi_class="multinomial").fit(PCs,y)
```


```python
confusion_matrix(y,clf2.predict(PCs))
```




    array([[50,  0,  0],
           [ 0, 47,  3],
           [ 0,  2, 48]])



- 임의로 변수 2개 만을 뽑아내여 분석한 경우 모델의 퍼포먼스가 하락함.


```python
clf = LogisticRegression(solver='sag', max_iter=1000, random_state=0,
                             multi_class="multinomial").fit(X2[:,0:2], y)
```


```python
confusion_matrix(y, clf.predict(X2[:,0:2]))
```




    array([[50,  0,  0],
           [ 0, 37, 13],
           [ 0, 14, 36]])



- 위와 같이, 차원축소를 통하여 모델의 복잡성을 줄이는 동시에 최대한 많은 정보를 활용하여 분석할 수 있음.
