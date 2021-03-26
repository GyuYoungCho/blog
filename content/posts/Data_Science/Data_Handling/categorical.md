---
title: "범주형 변수 처리"
date: 2021-03-27T04:11:41+09:00
draft: false
categories : ["Data_Science", "Data_Handling"]
---

```python
import pandas as pd
import os

df = pd.read_csv("car-good.csv")
```

```python
# 특징과 라벨 분리
X = df.drop('Class', axis = 1)
Y = df['Class']
```


```python
# 학습 데이터와 평가 데이터 분리
from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)
```

```python
# 문자 라벨을 숫자로 치환 
Train_Y.replace({"negative":-1, "positive":1}, inplace = True)
Test_Y.replace({"negative":-1, "positive":1}, inplace = True)
```

```python
# 자세한 범주형 변수 판별 => 모든 변수가 범주형임을 확인
for col in Train_X.columns:
    print(col, len(Train_X[col].unique()))
```

    Buying 4
    Maint 4
    Doors 3
    Persons 2
    Lug_boot 3
    Safety 3
    

#### 더미화를 이용한 범주 변수 처리


```python
Train_X = Train_X.astype(str) # 모든 변수가 범주이므로, 더미화를 위해 전부 string 타입으로 변환
```


```python
from feature_engine.categorical_encoders import OneHotCategoricalEncoder as OHE
dummy_model = OHE(variables = Train_X.columns.tolist(),
                 drop_last = True)

dummy_model.fit(Train_X)

d_Train_X = dummy_model.transform(Train_X)
d_Test_X = dummy_model.transform(Test_X)
```

      res_values = method(rvalues)
    


```python
# 더미화를 한 뒤의 모델 테스트
from sklearn.neighbors import KNeighborsClassifier as KNN
model = KNN().fit(d_Train_X, Train_Y)
pred_Y = model.predict(d_Test_X)

from sklearn.metrics import f1_score
f1_score(Test_Y, pred_Y)
```


    0.0



#### 연속형 변수로 치환


```python
Train_df = pd.concat([Train_X, Train_Y], axis = 1)
for col in Train_X.columns: # 보통은 범주 변수만 순회
    temp_dict = Train_df.groupby(col)['Class'].mean().to_dict() # col에 따른 Class의 평균을 나타내는 사전 (replace를 쓰기 위해, 사전으로 만듦)
    Train_df[col] = Train_df[col].replace(temp_dict) # 변수 치환    
    Test_X[col] = Test_X[col].astype(str).replace(temp_dict) # 테스트 데이터도 같이 치환해줘야 함 (나중에 활용하기 위해서는 저장도 필요)
```

    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    


```python
Train_df.head()
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
      <th>Buying</th>
      <th>Maint</th>
      <th>Doors</th>
      <th>Persons</th>
      <th>Lug_boot</th>
      <th>Safety</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>810</th>
      <td>-0.809524</td>
      <td>-0.82716</td>
      <td>-0.913462</td>
      <td>-1.0</td>
      <td>-0.921951</td>
      <td>-1.000000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>471</th>
      <td>-0.925466</td>
      <td>-1.00000</td>
      <td>-0.935185</td>
      <td>-1.0</td>
      <td>-0.926267</td>
      <td>-1.000000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>381</th>
      <td>-1.000000</td>
      <td>-0.82716</td>
      <td>-0.913462</td>
      <td>-1.0</td>
      <td>-0.926267</td>
      <td>-1.000000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>-1.000000</td>
      <td>-1.00000</td>
      <td>-0.946429</td>
      <td>-1.0</td>
      <td>-0.946903</td>
      <td>-0.869159</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>637</th>
      <td>-0.925466</td>
      <td>-0.82716</td>
      <td>-0.935185</td>
      <td>-1.0</td>
      <td>-0.946903</td>
      <td>-0.924171</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
Train_X = Train_df.drop('Class', axis = 1)
Train_Y = Train_df['Class']
```


```python
# 치환한 뒤의 모델 테스트
model = KNN().fit(Train_X, Train_Y)
pred_Y = model.predict(Test_X)

f1_score(Test_Y, pred_Y)


# 라벨을 고려한 전처리이므로 더미화보다 좋은 결과가 나왔음 => 차원도 줄고 성능 상에 이점이 있으나, 
```




    0.20000000000000004