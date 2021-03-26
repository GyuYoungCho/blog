---
title: "RandomForest"
date: 2021-03-26T11:35:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---



```python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```


```python
# 데이터 불러오기
data = pd.read_csv("./data/otto_train.csv") # Product Category
data.head() # 데이터 확인
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
      <th>id</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>...</th>
      <th>feat_85</th>
      <th>feat_86</th>
      <th>feat_87</th>
      <th>feat_88</th>
      <th>feat_89</th>
      <th>feat_90</th>
      <th>feat_91</th>
      <th>feat_92</th>
      <th>feat_93</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
  </tbody>
</table>

</div>




```python
'''
id: 고유 아이디
feat_1 ~ feat_93: 설명변수
target: 타겟변수 (1~9)
'''
```

    '\nid: 고유 아이디\nfeat_1 ~ feat_93: 설명변수\ntarget: 타겟변수 (1~9)\n'




```python
nCar = data.shape[0] # 데이터 개수
nVar = data.shape[1] # 변수 개수
print('nCar: %d' % nCar, 'nVar: %d' % nVar )
```

    nCar: 61878 nVar: 95
    




```python
data = data.drop(['id'], axis = 1) # id 제거
```

`타겟 변수의 문자열을 숫자로 변환


```python
mapping_dict = {"Class_1": 1,
                "Class_2": 2,
                "Class_3": 3,
                "Class_4": 4,
                "Class_5": 5,
                "Class_6": 6,
                "Class_7": 7,
                "Class_8": 8,
                "Class_9": 9}
after_mapping_target = data['target'].apply(lambda x: mapping_dict[x])
```



설명변수와 타겟변수를 분리, 학습데이터와 평가데이터 분리


```python
feature_columns = list(data.columns.difference(['target'])) 
X = data[feature_columns] 
y = after_mapping_target 
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42) 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 
```

    (49502, 93) (12376, 93) (49502,) (12376,)
    

- 학습 데이터를 랜덤포레스트 모형에 적합 후 평가 데이터로 검증


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
random_forest_model1 = RandomForestClassifier(n_estimators = 20,
                                             max_depth = 5, 
                                             random_state = 42) 
model1 = random_forest_model1.fit(train_x, train_y) 
predict1 = model1.predict(test_x) 
print("Accuracy: %.2f" % (accuracy_score(test_y, predict1) * 100), "%") 
```

    Accuracy: 60.16 %
    
**estimators 증가**

```python
random_forest_model2 = RandomForestClassifier(n_estimators = 300,
                                             max_depth = 5, 
                                             random_state = 42) 
model2 = random_forest_model2.fit(train_x, train_y) 
predict2 = model2.predict(test_x) 
print("Accuracy: %.2f" % (accuracy_score(test_y, predict2) * 100), "%")
```

    Accuracy: 61.73 %
    

**트리의 깊이**


```python
random_forest_model3 = RandomForestClassifier(n_estimators = 300,
                                             max_depth = 20, 
                                             random_state = 42) 
model3 = random_forest_model3.fit(train_x, train_y)
predict3 = model3.predict(test_x) 
print("Accuracy: %.2f" % (accuracy_score(test_y, predict3) * 100), "%")
```

    Accuracy: 78.09 %
    

**트리의 깊이를 최대**


```python
random_forest_model4 = RandomForestClassifier(n_estimators = 300,
                                             max_depth = 100,
                                             random_state = 42) 
model4 = random_forest_model4.fit(train_x, train_y) 
predict4 = model4.predict(test_x) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, predict4) * 100), "%")
```

    Accuracy: 81.23 %


