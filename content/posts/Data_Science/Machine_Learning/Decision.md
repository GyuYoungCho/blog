---
title: "Decision Tree"
date: 2021-03-26T02:15:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---

```python
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
```


```python
clf.predict([[1, 1]])
```


```python
from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()
```

#### 의사결정나무 구축 및 시각화

- 트리 구축


```python
clf=tree.DecisionTreeClassifier()
clf=clf.fit(iris.data,iris.target)
```

- 트리의 시각화


```python
dot_data=tree.export_graphviz(clf,out_file=None,
                             feature_names=iris.feature_names,
                            class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True
                             )
graph=graphviz.Source(dot_data)
```

![image](https://user-images.githubusercontent.com/49333349/112594910-f7bdc580-8e4c-11eb-8b7f-98cc718dbd8a.png)\

- Confusion Matrix 구하기


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(iris.target,clf.predict(iris.data))
```




    array([[50,  0,  0],
           [ 0, 50,  0],
           [ 0,  0, 50]])



