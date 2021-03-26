---
title: "회귀 분석을 이용한 앨범 판매량 분석 및 예측"
date: 2021-03-25T00:11:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')

sales = pd.read_csv('Album_sales_2.txt',sep='\t')
sales
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
      <th>adverts</th>
      <th>sales</th>
      <th>airplay</th>
      <th>attract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.256</td>
      <td>330</td>
      <td>43</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>985.685</td>
      <td>120</td>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1445.563</td>
      <td>360</td>
      <td>35</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1188.193</td>
      <td>270</td>
      <td>33</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>574.513</td>
      <td>220</td>
      <td>44</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>910.851</td>
      <td>190</td>
      <td>26</td>
      <td>7</td>
    </tr>
    <tr>
      <th>196</th>
      <td>888.569</td>
      <td>240</td>
      <td>14</td>
      <td>6</td>
    </tr>
    <tr>
      <th>197</th>
      <td>800.615</td>
      <td>250</td>
      <td>34</td>
      <td>6</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1500.000</td>
      <td>230</td>
      <td>11</td>
      <td>8</td>
    </tr>
    <tr>
      <th>199</th>
      <td>785.694</td>
      <td>110</td>
      <td>20</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

- advers : 광고비
- sales: 판매량
- airplay : 음반 출시전 한 주 동안 노래들이 라디오1에 방송된 횟수
- attract : 밴드 매력(0~10)

### 1. EDA


```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   adverts  200 non-null    float64
     1   sales    200 non-null    int64  
     2   airplay  200 non-null    int64  
     3   attract  200 non-null    int64  
    dtypes: float64(1), int64(3)
    memory usage: 6.4 KB
    

```python
sales.describe()
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
      <th>adverts</th>
      <th>sales</th>
      <th>airplay</th>
      <th>attract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>614.412255</td>
      <td>193.200000</td>
      <td>27.500000</td>
      <td>6.77000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>485.655208</td>
      <td>80.698957</td>
      <td>12.269585</td>
      <td>1.39529</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.104000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>215.917750</td>
      <td>137.500000</td>
      <td>19.750000</td>
      <td>6.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>531.916000</td>
      <td>200.000000</td>
      <td>28.000000</td>
      <td>7.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>911.225500</td>
      <td>250.000000</td>
      <td>36.000000</td>
      <td>8.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2271.860000</td>
      <td>360.000000</td>
      <td>63.000000</td>
      <td>10.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.style.use('seaborn-pastel')
plt.figure(figsize=(10,10))
for i,j in enumerate(sales.columns):
    plt.subplot(2,2,i+1)
    sns.histplot(sales[j])
```

![image](https://user-images.githubusercontent.com/49333349/112409090-3758b480-8d5c-11eb-9db5-51806fea4848.png)



```python
sns.pairplot(sales)
```

    <seaborn.axisgrid.PairGrid at 0x1e5f655a940>


![image](https://user-images.githubusercontent.com/49333349/112409173-58210a00-8d5c-11eb-9c76-568a010e95a2.png)

```python
sns.heatmap(sales.corr(),annot=True,cmap='Blues')
```
    <AxesSubplot:>

![image](https://user-images.githubusercontent.com/49333349/112409209-6e2eca80-8d5c-11eb-9eca-16df3aacf646.png)

`시각화를 통해 보았을 때 advert가 치우쳐서 변수 변환이 필요할 수 있다는 점 빼고 큰 특징은 보이지 않는다.

상관성이 좀 있는 편인데 일단 회귀 분석을 진행하면서 결과를 살펴봐야겠다.

---
<br>

### 2. train, test로 분리 후 다중회귀분석 결과 해석

```python
x_train, x_test, y_train, y_test = train_test_split(sales.iloc[:,[0,2,3]], sales['sales'],)
```

상수항 넣었을 때

```python
x_data = sm.add_constant(x_train, has_constant = "add")
```

```python
# const넣음
model = sm.OLS(y_train,x_data)
result = model.fit()
result.summary()
```

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>sales</td>      <th>  R-squared:         </th> <td>   0.651</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.644</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   90.71</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 09 Oct 2020</td> <th>  Prob (F-statistic):</th> <td>3.44e-33</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:24:41</td>     <th>  Log-Likelihood:    </th> <td> -789.91</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   150</td>      <th>  AIC:               </th> <td>   1588.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   146</td>      <th>  BIC:               </th> <td>   1600.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>  -36.6149</td> <td>   21.619</td> <td>   -1.694</td> <td> 0.092</td> <td>  -79.341</td> <td>    6.111</td>
</tr>
<tr>
  <th>adverts</th> <td>    0.0825</td> <td>    0.008</td> <td>   10.032</td> <td> 0.000</td> <td>    0.066</td> <td>    0.099</td>
</tr>
<tr>
  <th>airplay</th> <td>    3.4240</td> <td>    0.314</td> <td>   10.897</td> <td> 0.000</td> <td>    2.803</td> <td>    4.045</td>
</tr>
<tr>
  <th>attract</th> <td>   12.5468</td> <td>    2.997</td> <td>    4.187</td> <td> 0.000</td> <td>    6.624</td> <td>   18.470</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.238</td> <th>  Durbin-Watson:     </th> <td>   1.725</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.888</td> <th>  Jarque-Bera (JB):  </th> <td>   0.189</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.086</td> <th>  Prob(JB):          </th> <td>   0.910</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.972</td> <th>  Cond. No.          </th> <td>4.41e+03</td>
</tr>
</table><br/><br/>




```python
y_pred = result.predict(ols.add_constant(x_test, has_constant = "add"))
mean_squared_error(y_pred,y_test)
```

    2128.1345418881824


```python
# 안 넣음
model2 = sm.OLS(y_train,x_train)
result2 = model2.fit()
result2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>sales</td>      <th>  R-squared (uncentered):</th>      <td>   0.950</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.949</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   928.3</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Oct 2020</td> <th>  Prob (F-statistic):</th>          <td>2.75e-95</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:09:26</td>     <th>  Log-Likelihood:    </th>          <td> -791.37</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   150</td>      <th>  AIC:               </th>          <td>   1589.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   147</td>      <th>  BIC:               </th>          <td>   1598.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>              <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>adverts</th> <td>    0.0797</td> <td>    0.008</td> <td>    9.830</td> <td> 0.000</td> <td>    0.064</td> <td>    0.096</td>
</tr>
<tr>
  <th>airplay</th> <td>    3.3006</td> <td>    0.308</td> <td>   10.731</td> <td> 0.000</td> <td>    2.693</td> <td>    3.908</td>
</tr>
<tr>
  <th>attract</th> <td>    8.1136</td> <td>    1.468</td> <td>    5.525</td> <td> 0.000</td> <td>    5.212</td> <td>   11.016</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.694</td> <th>  Durbin-Watson:     </th> <td>   1.762</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.707</td> <th>  Jarque-Bera (JB):  </th> <td>   0.587</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.153</td> <th>  Prob(JB):          </th> <td>   0.745</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.990</td> <th>  Cond. No.          </th> <td>    300.</td>
</tr>
</table><br/>

aic는 비슷하나 R-squared 면에서 상수항이 없는게 나음


```python
y_pred = result2.predict(x_test)
mean_squared_error(y_pred,y_test)
```

    2093.0726936877363


```python
plt.plot(y_test.values)
plt.plot(y_pred.values)
```

![image](https://user-images.githubusercontent.com/49333349/112411492-56f1dc00-8d60-11eb-8310-52e8e769ee9d.png)


**VIF**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor 
[variance_inflation_factor(x_train.values,i) for i in range(3)]
```

    [2.6632477766856404, 5.761240629832455, 6.855611086457372]

대체로 크게 나오는 편은 아니다.

변수가 적어 모든 변수 조합에서 Variance와 mse를 살펴 봄


```python
from itertools import combinations
joint = []
for i in range(1,4):
    k = combinations(x_train.columns,i)
    for j in k:
        joint.append(j)
```


    [('adverts',),
     ('airplay',),
     ('attract',),
     ('adverts', 'airplay'),
     ('adverts', 'attract'),
     ('airplay', 'attract'),
     ('adverts', 'airplay', 'attract')]




```python
stepw = []
for i in joint:
    model = sm.OLS(y_train,x_train[list(i)])
    result = model.fit()
    y_pred = result.predict(x_test[list(i)])
    stepw.append([list(i),result.aic,mean_squared_error(y_pred,y_test)])
```


```python
stepw_frame = pd.DataFrame(stepw, columns=['Variables','AIC','MSE'])
stepw_frame.set_index('Variables',inplace=True)
stepw_frame
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
      <th>AIC</th>
      <th>MSE</th>
    </tr>
    <tr>
      <th>Variables</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[adverts]</th>
      <td>1832.620857</td>
      <td>9775.286334</td>
    </tr>
    <tr>
      <th>[airplay]</th>
      <td>1716.242475</td>
      <td>5246.164309</td>
    </tr>
    <tr>
      <th>[attract]</th>
      <td>1727.874967</td>
      <td>6423.816111</td>
    </tr>
    <tr>
      <th>[adverts, airplay]</th>
      <td>1615.042139</td>
      <td>2447.023190</td>
    </tr>
    <tr>
      <th>[adverts, attract]</th>
      <td>1673.510244</td>
      <td>3376.763066</td>
    </tr>
    <tr>
      <th>[airplay, attract]</th>
      <td>1662.523096</td>
      <td>4237.857648</td>
    </tr>
    <tr>
      <th>[adverts, airplay, attract]</th>
      <td>1588.737818</td>
      <td>2093.072694</td>
    </tr>
  </tbody>
</table>
</div>



3개 다 선택했을 때 값이 가장 적게 나온다.

<br>

**다중회귀분석의 매개변수별 신뢰구간**


```python
result2.conf_int()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>adverts</th>
      <td>0.063717</td>
      <td>0.095782</td>
    </tr>
    <tr>
      <th>airplay</th>
      <td>2.692710</td>
      <td>3.908404</td>
    </tr>
    <tr>
      <th>attract</th>
      <td>5.211536</td>
      <td>11.015580</td>
    </tr>
  </tbody>
</table>
</div>


### 3. 독립성, 정규성 가정 평가


```python
# 독립성
sns.regplot(result2.fittedvalues, result2.resid, lowess=True, line_kws={'color': 'red'})
```

![image](https://user-images.githubusercontent.com/49333349/112411733-be0f9080-8d60-11eb-9778-7fce3d55362c.png)



```python
#정규성
stats.probplot(result2.resid,plot=plt);
```

![image](https://user-images.githubusercontent.com/49333349/112411971-121a7500-8d61-11eb-91b4-df53ea4715bf.png)


```python
stats.shapiro(result2.resid)
```

    (0.9950959086418152, 0.8982971906661987)

- 독립성과 정규성을 만족하는 편이다.
- summary에서 durbin-watson 값이 1.5~2.5 안에 있어야 정상이라고 함

### 4. 표준화잔차, cook's distance, DFBeta, 공분산비 보기

- 표준화잔차(standard_resid) : 잔차를 표준오차로 나눈 값
- cook's distance : 관측치가 제거되었을 때 모수 추정값들의 (동시적) 변화량 척도
- DFBeta : 관측값이 각각의 회귀계수에 미치는 영향력을 측정
- DFFITS : 관측치가 예측값에서 보유하고 있는 영향력을 측정

```python
infl = result2.get_influence()
sm_fr = infl.summary_frame()
sm_fr
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
      <th>dfb_adverts</th>
      <th>dfb_airplay</th>
      <th>dfb_attract</th>
      <th>cooks_d</th>
      <th>standard_resid</th>
      <th>hat_diag</th>
      <th>dffits_internal</th>
      <th>student_resid</th>
      <th>dffits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>-0.000441</td>
      <td>-0.009346</td>
      <td>0.000516</td>
      <td>0.000159</td>
      <td>-0.238252</td>
      <td>0.008315</td>
      <td>-0.021816</td>
      <td>-0.237486</td>
      <td>-0.021746</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-0.005728</td>
      <td>0.022027</td>
      <td>-0.020366</td>
      <td>0.000221</td>
      <td>-0.152693</td>
      <td>0.027656</td>
      <td>-0.025751</td>
      <td>-0.152185</td>
      <td>-0.025666</td>
    </tr>
    <tr>
      <th>117</th>
      <td>-0.004076</td>
      <td>0.000292</td>
      <td>-0.001237</td>
      <td>0.000026</td>
      <td>-0.133886</td>
      <td>0.004372</td>
      <td>-0.008872</td>
      <td>-0.133438</td>
      <td>-0.008842</td>
    </tr>
    <tr>
      <th>93</th>
      <td>-0.054948</td>
      <td>-0.212492</td>
      <td>0.243871</td>
      <td>0.020558</td>
      <td>1.247608</td>
      <td>0.038112</td>
      <td>0.248342</td>
      <td>1.249992</td>
      <td>0.248816</td>
    </tr>
    <tr>
      <th>64</th>
      <td>-0.029928</td>
      <td>-0.040401</td>
      <td>0.068245</td>
      <td>0.001846</td>
      <td>0.489364</td>
      <td>0.022606</td>
      <td>0.074423</td>
      <td>0.488095</td>
      <td>0.074230</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153</th>
      <td>-0.050052</td>
      <td>0.055071</td>
      <td>-0.006892</td>
      <td>0.002166</td>
      <td>0.452345</td>
      <td>0.030779</td>
      <td>0.080610</td>
      <td>0.451118</td>
      <td>0.080391</td>
    </tr>
    <tr>
      <th>130</th>
      <td>0.002361</td>
      <td>0.053852</td>
      <td>-0.030313</td>
      <td>0.001554</td>
      <td>0.697670</td>
      <td>0.009485</td>
      <td>0.068270</td>
      <td>0.696447</td>
      <td>0.068151</td>
    </tr>
    <tr>
      <th>104</th>
      <td>-0.048983</td>
      <td>0.167837</td>
      <td>-0.092858</td>
      <td>0.010958</td>
      <td>0.829723</td>
      <td>0.045577</td>
      <td>0.181316</td>
      <td>0.828839</td>
      <td>0.181123</td>
    </tr>
    <tr>
      <th>161</th>
      <td>0.094157</td>
      <td>0.025861</td>
      <td>-0.098308</td>
      <td>0.005490</td>
      <td>-1.127330</td>
      <td>0.012794</td>
      <td>-0.128339</td>
      <td>-1.128377</td>
      <td>-0.128458</td>
    </tr>
    <tr>
      <th>183</th>
      <td>-0.028555</td>
      <td>-0.003881</td>
      <td>0.015144</td>
      <td>0.000299</td>
      <td>-0.089645</td>
      <td>0.100417</td>
      <td>-0.029951</td>
      <td>-0.089342</td>
      <td>-0.029850</td>
    </tr>
  </tbody>
</table>
</div>


```python
album_sales_diagnostics = pd.concat([x_train,y_train],axis=1)

album_sales_diagnostics['cook'] = infl.cooks_distance[0]
album_sales_diagnostics['resid_std'] = infl.resid_studentized
album_sales_diagnostics[['dfbeta_adverts','dfbeta_airplay','dfbeta_attract']] = infl.dfbeta
album_sales_diagnostics['covariance'] = infl.cov_ratio
album_sales_diagnostics.head()
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
      <th>adverts</th>
      <th>airplay</th>
      <th>attract</th>
      <th>sales</th>
      <th>cook</th>
      <th>resid_std</th>
      <th>dfbeta_adverts</th>
      <th>dfbeta_airplay</th>
      <th>dfbeta_attract</th>
      <th>covariance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>656.137</td>
      <td>34</td>
      <td>7</td>
      <td>210</td>
      <td>0.000159</td>
      <td>-0.238252</td>
      <td>-0.000004</td>
      <td>-0.002884</td>
      <td>0.000760</td>
      <td>1.028055</td>
    </tr>
    <tr>
      <th>33</th>
      <td>759.862</td>
      <td>6</td>
      <td>7</td>
      <td>130</td>
      <td>0.000221</td>
      <td>-0.152693</td>
      <td>-0.000047</td>
      <td>0.006798</td>
      <td>-0.030006</td>
      <td>1.049221</td>
    </tr>
    <tr>
      <th>117</th>
      <td>624.538</td>
      <td>20</td>
      <td>5</td>
      <td>150</td>
      <td>0.000026</td>
      <td>-0.133886</td>
      <td>-0.000033</td>
      <td>0.000090</td>
      <td>-0.001823</td>
      <td>1.024796</td>
    </tr>
    <tr>
      <th>93</th>
      <td>268.598</td>
      <td>1</td>
      <td>7</td>
      <td>140</td>
      <td>0.020558</td>
      <td>1.247608</td>
      <td>-0.000445</td>
      <td>-0.065233</td>
      <td>0.357433</td>
      <td>1.027779</td>
    </tr>
    <tr>
      <th>64</th>
      <td>391.749</td>
      <td>22</td>
      <td>9</td>
      <td>200</td>
      <td>0.001846</td>
      <td>0.489364</td>
      <td>-0.000243</td>
      <td>-0.012459</td>
      <td>0.100476</td>
      <td>1.039201</td>
    </tr>
  </tbody>
</table>
</div>


**영향력이 큰 사례**

cook distance가 큰 순서대로 정렬
- 4/표본수 이상이면 영향력이 크다고 할 수 있음.

```python
album.resid_std = np.abs(album.resid_std)
album.sort_values(by = ['cook','resid_std'],ascending=[False, False])
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
      <th>adverts</th>
      <th>airplay</th>
      <th>attract</th>
      <th>sales</th>
      <th>cook</th>
      <th>resid_std</th>
      <th>dfbeta_adverts</th>
      <th>dfbeta_airplay</th>
      <th>dfbeta_attract</th>
      <th>covariance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>168</th>
      <td>145.585</td>
      <td>42</td>
      <td>8</td>
      <td>360</td>
      <td>0.075704</td>
      <td>3.067079</td>
      <td>-0.002579</td>
      <td>0.071664</td>
      <td>0.107867</td>
      <td>0.857218</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10.256</td>
      <td>43</td>
      <td>10</td>
      <td>330</td>
      <td>0.065682</td>
      <td>2.262619</td>
      <td>-0.002828</td>
      <td>0.025215</td>
      <td>0.302940</td>
      <td>0.953043</td>
    </tr>
    <tr>
      <th>118</th>
      <td>912.349</td>
      <td>57</td>
      <td>6</td>
      <td>230</td>
      <td>0.053361</td>
      <td>1.709781</td>
      <td>-0.000618</td>
      <td>-0.111819</td>
      <td>0.430083</td>
      <td>1.013622</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1000.000</td>
      <td>5</td>
      <td>7</td>
      <td>250</td>
      <td>0.050402</td>
      <td>2.064131</td>
      <td>0.001338</td>
      <td>-0.098943</td>
      <td>0.371103</td>
      <td>0.967650</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1542.329</td>
      <td>33</td>
      <td>8</td>
      <td>190</td>
      <td>0.050391</td>
      <td>2.267631</td>
      <td>-0.002619</td>
      <td>0.004308</td>
      <td>0.094568</td>
      <td>0.944246</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>15.313</td>
      <td>22</td>
      <td>5</td>
      <td>110</td>
      <td>0.000026</td>
      <td>0.092506</td>
      <td>0.000056</td>
      <td>-0.000595</td>
      <td>-0.005664</td>
      <td>1.029915</td>
    </tr>
    <tr>
      <th>117</th>
      <td>624.538</td>
      <td>20</td>
      <td>5</td>
      <td>150</td>
      <td>0.000026</td>
      <td>0.133886</td>
      <td>-0.000033</td>
      <td>0.000090</td>
      <td>-0.001823</td>
      <td>1.024796</td>
    </tr>
    <tr>
      <th>179</th>
      <td>26.598</td>
      <td>47</td>
      <td>8</td>
      <td>220</td>
      <td>0.000025</td>
      <td>0.045893</td>
      <td>0.000048</td>
      <td>-0.001569</td>
      <td>-0.000459</td>
      <td>1.056691</td>
    </tr>
    <tr>
      <th>141</th>
      <td>893.355</td>
      <td>26</td>
      <td>6</td>
      <td>210</td>
      <td>0.000023</td>
      <td>0.089516</td>
      <td>0.000044</td>
      <td>0.000217</td>
      <td>-0.001188</td>
      <td>1.029495</td>
    </tr>
    <tr>
      <th>109</th>
      <td>102.568</td>
      <td>22</td>
      <td>7</td>
      <td>140</td>
      <td>0.000013</td>
      <td>0.050878</td>
      <td>-0.000035</td>
      <td>-0.000455</td>
      <td>0.007294</td>
      <td>1.036476</td>
    </tr>
  </tbody>
</table>
</div>




데이터 출처 : [SpringSchool](https://belik.userpage.fu-berlin.de/springschool/)