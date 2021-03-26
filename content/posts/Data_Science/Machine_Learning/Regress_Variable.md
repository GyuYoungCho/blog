---
title: "변수 선택법, 회귀 계수 축소"
date: 2021-03-27T03:11:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---




```python
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import time
ploan = pd.read_csv("./data/Personal Loan.csv")
```


```python
ploan_processed = ploan.dropna().drop(['ID','ZIP Code'], axis=1, inplace=False)
ploan_processed = sm.add_constant(ploan_processed, has_constant='add')
```



- 설명변수(X), 타켓변수(Y) 분리 및 학습데이터와 평가데이터

```python
feature_columns = list(ploan_processed.columns.difference(["Personal Loan"]))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan'] # 대출여부: 1 or 0
```


```python
train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y,train_size=0.7,test_size=0.3,random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
```

    (1750, 12) (750, 12) (1750,) (750,)
    

# 로지스틱회귀모형 모델링 y = f(x)


```python
model = sm.Logit(train_y, train_x)
results = model.fit(method='newton')
```

    Optimization terminated successfully.
             Current function value: 0.131055
             Iterations 9
    


```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>Personal Loan</td>  <th>  No. Observations:  </th>   <td>  1750</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  1738</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>    11</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 23 Aug 2019</td> <th>  Pseudo R-squ.:     </th>   <td>0.6030</td>  
</tr>
<tr>
  <th>Time:</th>              <td>14:58:19</td>     <th>  Log-Likelihood:    </th>  <td> -229.35</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th>  <td> -577.63</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>2.927e-142</td>
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Age</th>                <td>    0.0245</td> <td>    0.102</td> <td>    0.240</td> <td> 0.810</td> <td>   -0.175</td> <td>    0.224</td>
</tr>
<tr>
  <th>CCAvg</th>              <td>    0.0985</td> <td>    0.063</td> <td>    1.562</td> <td> 0.118</td> <td>   -0.025</td> <td>    0.222</td>
</tr>
<tr>
  <th>CD Account</th>         <td>    4.3726</td> <td>    0.568</td> <td>    7.703</td> <td> 0.000</td> <td>    3.260</td> <td>    5.485</td>
</tr>
<tr>
  <th>CreditCard</th>         <td>   -1.2374</td> <td>    0.337</td> <td>   -3.667</td> <td> 0.000</td> <td>   -1.899</td> <td>   -0.576</td>
</tr>
<tr>
  <th>Education</th>          <td>    1.5203</td> <td>    0.190</td> <td>    7.999</td> <td> 0.000</td> <td>    1.148</td> <td>    1.893</td>
</tr>
<tr>
  <th>Experience</th>         <td>   -0.0070</td> <td>    0.102</td> <td>   -0.069</td> <td> 0.945</td> <td>   -0.206</td> <td>    0.192</td>
</tr>
<tr>
  <th>Family</th>             <td>    0.7579</td> <td>    0.128</td> <td>    5.914</td> <td> 0.000</td> <td>    0.507</td> <td>    1.009</td>
</tr>
<tr>
  <th>Income</th>             <td>    0.0547</td> <td>    0.004</td> <td>   12.659</td> <td> 0.000</td> <td>    0.046</td> <td>    0.063</td>
</tr>
<tr>
  <th>Mortgage</th>           <td>   -0.0001</td> <td>    0.001</td> <td>   -0.144</td> <td> 0.885</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>Online</th>             <td>   -0.4407</td> <td>    0.263</td> <td>   -1.674</td> <td> 0.094</td> <td>   -0.957</td> <td>    0.075</td>
</tr>
<tr>
  <th>Securities Account</th> <td>   -1.8520</td> <td>    0.561</td> <td>   -3.299</td> <td> 0.001</td> <td>   -2.952</td> <td>   -0.752</td>
</tr>
<tr>
  <th>const</th>              <td>  -13.9203</td> <td>    2.773</td> <td>   -5.021</td> <td> 0.000</td> <td>  -19.354</td> <td>   -8.486</td>
</tr>
</table>




```python
# performance measure
print("model AIC: ","{:.5f}".format(results.aic))
```

    model AIC:  482.69329
    


```python
results.params
```




    Age                    0.024471
    CCAvg                  0.098468
    CD Account             4.372577
    CreditCard            -1.237447
    Education              1.520329
    Experience            -0.007032
    Family                 0.757911
    Income                 0.054695
    Mortgage              -0.000133
    Online                -0.440746
    Securities Account    -1.852006
    const                -13.920298
    dtype: float64




```python
## 나이가 한살 많을수록록 대출할 확률이 1.024 높다.
## 수입이 1단위 높을소룩 대출할 확률이 1.05배 높다 
## 가족 구성원수가 1많을수록 대출할 확률이 2.13배 높다
## 경력이 1단위 높을수록 대출할 확률이 0.99배 높다(귀무가설 채택)
# Experience,  Mortgage는 제외할 필요성이 있어보임
np.exp(results.params)
```




    Age                   1.024773e+00
    CCAvg                 1.103479e+00
    CD Account            7.924761e+01
    CreditCard            2.901239e-01
    Education             4.573729e+00
    Experience            9.929928e-01
    Family                2.133814e+00
    Income                1.056218e+00
    Mortgage              9.998665e-01
    Online                6.435563e-01
    Securities Account    1.569221e-01
    const                 9.005163e-07
    dtype: float64




```python
pred_y = results.predict(test_x)
```


```python
def cut_off(y,threshold):
    Y = y.copy() # copy함수를 사용하여 이전의 y값이 변화지 않게 함
    Y[Y>threshold]=1
    Y[Y<=threshold]=0
    return(Y.astype(int))

pred_Y = cut_off(pred_y,0.5)
```

```python
cfmat = confusion_matrix(test_y,pred_Y)
print(cfmat)
```

    [[661  12]
     [ 28  49]]
    


```python
(cfmat[0,0]+cfmat[1,1])/np.sum(cfmat) ## accuracy
```




    0.9466666666666667




```python
def acc(cfmat) :
    acc=(cfmat[0,0]+cfmat[1,1])/np.sum(cfmat) ## accuracy
    return(acc)
```

#### 임계값(cut-off)에 따른 성능지표 비교



```python
threshold = np.arange(0,1,0.1)
table = pd.DataFrame(columns=['ACC'])
for i in threshold:
    pred_Y = cut_off(pred_y,i)
    cfmat = confusion_matrix(test_y, pred_Y)
    table.loc[i] = acc(cfmat)
table.index.name='threshold'
table.columns.name='performance'
table
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
      <th>performance</th>
      <th>ACC</th>
    </tr>
    <tr>
      <th>threshold</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.102667</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.908000</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.922667</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.934667</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.949333</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.941333</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.937333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sklearn ROC 패키지 제공
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=1)

# Print ROC curve
plt.plot(fpr,tpr)

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)


```

    AUC: 0.9463923891858513
    

![image](https://user-images.githubusercontent.com/49333349/112686056-c1b62b00-8eb8-11eb-8f93-676518db9321.png)


```python
feature_columns = list(ploan_processed.columns.difference(["Personal Loan","Experience",  "Mortgage"]))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan'] # 대출여부: 1 or 0
```


```python
train_x2, test_x2, train_y, test_y = train_test_split(X, y, stratify=y,train_size=0.7,test_size=0.3,random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
```

    (1750, 12) (750, 12) (1750,) (750,)
    


```python
model = sm.Logit(train_y, train_x2)
results2 = model.fit(method='newton')
```

    Optimization terminated successfully.
             Current function value: 0.131062
             Iterations 9
    


```python
results2.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>Personal Loan</td>  <th>  No. Observations:  </th>   <td>  1750</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  1740</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     9</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 23 Aug 2019</td> <th>  Pseudo R-squ.:     </th>   <td>0.6029</td>  
</tr>
<tr>
  <th>Time:</th>              <td>14:58:19</td>     <th>  Log-Likelihood:    </th>  <td> -229.36</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th>  <td> -577.63</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>3.817e-144</td>
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Age</th>                <td>    0.0174</td> <td>    0.011</td> <td>    1.569</td> <td> 0.117</td> <td>   -0.004</td> <td>    0.039</td>
</tr>
<tr>
  <th>CCAvg</th>              <td>    0.0997</td> <td>    0.062</td> <td>    1.596</td> <td> 0.111</td> <td>   -0.023</td> <td>    0.222</td>
</tr>
<tr>
  <th>CD Account</th>         <td>    4.3699</td> <td>    0.567</td> <td>    7.705</td> <td> 0.000</td> <td>    3.258</td> <td>    5.481</td>
</tr>
<tr>
  <th>CreditCard</th>         <td>   -1.2350</td> <td>    0.337</td> <td>   -3.668</td> <td> 0.000</td> <td>   -1.895</td> <td>   -0.575</td>
</tr>
<tr>
  <th>Education</th>          <td>    1.5249</td> <td>    0.187</td> <td>    8.156</td> <td> 0.000</td> <td>    1.158</td> <td>    1.891</td>
</tr>
<tr>
  <th>Family</th>             <td>    0.7572</td> <td>    0.127</td> <td>    5.948</td> <td> 0.000</td> <td>    0.508</td> <td>    1.007</td>
</tr>
<tr>
  <th>Income</th>             <td>    0.0546</td> <td>    0.004</td> <td>   12.833</td> <td> 0.000</td> <td>    0.046</td> <td>    0.063</td>
</tr>
<tr>
  <th>Online</th>             <td>   -0.4418</td> <td>    0.263</td> <td>   -1.678</td> <td> 0.093</td> <td>   -0.958</td> <td>    0.074</td>
</tr>
<tr>
  <th>Securities Account</th> <td>   -1.8526</td> <td>    0.561</td> <td>   -3.302</td> <td> 0.001</td> <td>   -2.952</td> <td>   -0.753</td>
</tr>
<tr>
  <th>const</th>              <td>  -13.7465</td> <td>    1.164</td> <td>  -11.814</td> <td> 0.000</td> <td>  -16.027</td> <td>  -11.466</td>
</tr>
</table>




```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>Personal Loan</td>  <th>  No. Observations:  </th>   <td>  1750</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  1738</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>    11</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 23 Aug 2019</td> <th>  Pseudo R-squ.:     </th>   <td>0.6030</td>  
</tr>
<tr>
  <th>Time:</th>              <td>14:58:19</td>     <th>  Log-Likelihood:    </th>  <td> -229.35</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th>  <td> -577.63</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>2.927e-142</td>
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Age</th>                <td>    0.0245</td> <td>    0.102</td> <td>    0.240</td> <td> 0.810</td> <td>   -0.175</td> <td>    0.224</td>
</tr>
<tr>
  <th>CCAvg</th>              <td>    0.0985</td> <td>    0.063</td> <td>    1.562</td> <td> 0.118</td> <td>   -0.025</td> <td>    0.222</td>
</tr>
<tr>
  <th>CD Account</th>         <td>    4.3726</td> <td>    0.568</td> <td>    7.703</td> <td> 0.000</td> <td>    3.260</td> <td>    5.485</td>
</tr>
<tr>
  <th>CreditCard</th>         <td>   -1.2374</td> <td>    0.337</td> <td>   -3.667</td> <td> 0.000</td> <td>   -1.899</td> <td>   -0.576</td>
</tr>
<tr>
  <th>Education</th>          <td>    1.5203</td> <td>    0.190</td> <td>    7.999</td> <td> 0.000</td> <td>    1.148</td> <td>    1.893</td>
</tr>
<tr>
  <th>Experience</th>         <td>   -0.0070</td> <td>    0.102</td> <td>   -0.069</td> <td> 0.945</td> <td>   -0.206</td> <td>    0.192</td>
</tr>
<tr>
  <th>Family</th>             <td>    0.7579</td> <td>    0.128</td> <td>    5.914</td> <td> 0.000</td> <td>    0.507</td> <td>    1.009</td>
</tr>
<tr>
  <th>Income</th>             <td>    0.0547</td> <td>    0.004</td> <td>   12.659</td> <td> 0.000</td> <td>    0.046</td> <td>    0.063</td>
</tr>
<tr>
  <th>Mortgage</th>           <td>   -0.0001</td> <td>    0.001</td> <td>   -0.144</td> <td> 0.885</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>Online</th>             <td>   -0.4407</td> <td>    0.263</td> <td>   -1.674</td> <td> 0.094</td> <td>   -0.957</td> <td>    0.075</td>
</tr>
<tr>
  <th>Securities Account</th> <td>   -1.8520</td> <td>    0.561</td> <td>   -3.299</td> <td> 0.001</td> <td>   -2.952</td> <td>   -0.752</td>
</tr>
<tr>
  <th>const</th>              <td>  -13.9203</td> <td>    2.773</td> <td>   -5.021</td> <td> 0.000</td> <td>  -19.354</td> <td>   -8.486</td>
</tr>
</table>




```python
pred_y = results2.predict(test_x2)
```


```python
pred_Y = cut_off(pred_y,0.5)
```




```python
cfmat = confusion_matrix(test_y,pred_Y)
print(cfmat)
```

    [[660  13]
     [ 29  48]]
    


```python
acc(cfmat) ## accuracy
```

    0.944


```python
threshold = np.arange(0,1,0.1)
table = pd.DataFrame(columns=['ACC'])
for i in threshold:
    pred_Y = cut_off(pred_y,i)
    cfmat = confusion_matrix(test_y, pred_Y)
    table.loc[i] =acc(cfmat)
table.index.name='threshold'
table.columns.name='performance'
table
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
      <th>performance</th>
      <th>ACC</th>
    </tr>
    <tr>
      <th>threshold</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.102667</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.908000</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.922667</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.932000</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.936000</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.944000</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.949333</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.941333</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.937333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sklearn ROC 패키지 제공
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=1)

# Print ROC curve
plt.plot(fpr,tpr)

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)

```

    AUC: 0.9465467667547905
    

---

## 변수선택법


```python
feature_columns = list(ploan_processed.columns.difference(["Personal Loan"]))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan'] # 대출여부: 1 or 0
```


```python
train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y,train_size=0.7,test_size=0.3,random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
```

    (1750, 12) (750, 12) (1750,) (750,)
    


```python
def processSubset(X,y, feature_set):
            model = sm.Logit(y,X[list(feature_set)])
            regr = model.fit()
            AIC = regr.aic
            return {"model":regr, "AIC":AIC}
        
'''
전진선택법
'''
def forward(X, y, predictors):
    # 데이터 변수들이 미리정의된 predictors에 있는지 없는지 확인 및 분류
    remaining_predictors = [p for p in X.columns.difference(['const']) if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(X=X, y= y, feature_set=predictors+[p]+['const']))
    # 데이터프레임으로 변환
    models = pd.DataFrame(results)

    # AIC가 가장 낮은 것을 선택
    best_model = models.loc[models['AIC'].argmin()] # index
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic))
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model[0] )
    return best_model

def forward_model(X,y):
    Fmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = []
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X,y=y,predictors=predictors)
        if i > 1:
            if Forward_result['AIC'] > Fmodel_before:
                break
        Fmodels.loc[i] = Forward_result
        predictors = Fmodels.loc[i]["model"].model.exog_names
        Fmodel_before = Fmodels.loc[i]["AIC"]
        predictors = [ k for k in predictors if k != 'const']
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return(Fmodels['model'][len(Fmodels['model'])])


'''
후진소거법
'''
def backward(X,y,predictors):
    tic = time.time()
    results = []
    
    # 데이터 변수들이 미리정의된 predictors 조합 확인
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(X=X, y= y,feature_set=list(combo)+['const']))
    models = pd.DataFrame(results)
    
    # 가장 낮은 AIC를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in",
          (toc - tic))
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model[0] )
    return best_model


def backward_model(X, y):
    Bmodels = pd.DataFrame(columns=["AIC", "model"], index = range(1,len(X.columns)))
    tic = time.time()
    predictors = X.columns.difference(['const'])
    Bmodel_before = processSubset(X,y,predictors)['AIC']
    while (len(predictors) > 1):
        Backward_result = backward(X=train_x, y= train_y, predictors = predictors)
        if Backward_result['AIC'] > Bmodel_before:
            break
        Bmodels.loc[len(predictors) - 1] = Backward_result
        predictors = Bmodels.loc[len(predictors) - 1]["model"].model.exog_names
        Bmodel_before = Backward_result['AIC']
        predictors = [ k for k in predictors if k != 'const']

    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Bmodels['model'].dropna().iloc[0])


'''
단계적 선택법
'''
def Stepwise_model(X,y):
    Stepmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    predictors = []
    Smodel_before = processSubset(X,y,predictors+['const'])['AIC']
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X, y=y, predictors=predictors) # constant added
        print('forward')
        Stepmodels.loc[i] = Forward_result
        predictors = Stepmodels.loc[i]["model"].model.exog_names
        predictors = [ k for k in predictors if k != 'const']
        Backward_result = backward(X=X, y=y, predictors=predictors)
        if Backward_result['AIC']< Forward_result['AIC']:
            Stepmodels.loc[i] = Backward_result
            predictors = Stepmodels.loc[i]["model"].model.exog_names
            Smodel_before = Stepmodels.loc[i]["AIC"]
            predictors = [ k for k in predictors if k != 'const']
            print('backward')
        if Stepmodels.loc[i]['AIC']> Smodel_before:
            break
        else:
            Smodel_before = Stepmodels.loc[i]["AIC"]
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Stepmodels['model'][len(Stepmodels['model'])])
```


```python
Forward_best_model = forward_model(X=train_x, y= train_y)
```
    


```python
Backward_best_model = backward_model(X=train_x,y=train_y)
```

   
```python
Stepwise_best_model = Stepwise_model(X=train_x,y=train_y)
```

    Total elapsed time: 0.9743940830230713 seconds.
    


```python
pred_y_full = results2.predict(test_x2) # full model
pred_y_forward = Forward_best_model.predict(test_x[Forward_best_model.model.exog_names])
pred_y_backward = Backward_best_model.predict(test_x[Backward_best_model.model.exog_names])
pred_y_stepwise = Stepwise_best_model.predict(test_x[Stepwise_best_model.model.exog_names])
```


```python
pred_Y_full= cut_off(pred_y_full,0.5)
pred_Y_forward = cut_off(pred_y_forward,0.5)
pred_Y_backward = cut_off(pred_y_backward,0.5)
pred_Y_stepwise = cut_off(pred_y_stepwise,0.5)
```


```python
cfmat_full = confusion_matrix(test_y, pred_Y_full)
cfmat_forward = confusion_matrix(test_y, pred_Y_forward)
cfmat_backward = confusion_matrix(test_y, pred_Y_backward)
cfmat_stepwise = confusion_matrix(test_y, pred_Y_stepwise)
```


```python
print(acc(cfmat_full))
print(acc(cfmat_forward))
print(acc(cfmat_backward))
print(acc(cfmat_stepwise))

```

    0.944
    0.944
    0.944
    0.944
    


```python
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_full, pos_label=1)
# Print ROC curve
plt.plot(fpr,tpr)
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
```

    AUC: 0.9465467667547905
    


![png](output_44_1.png)



```python
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_forward, pos_label=1)
# Print ROC curve
plt.plot(fpr,tpr)
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
```

    AUC: 0.9465467667547905
    


![png](output_45_1.png)



```python
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_backward, pos_label=1)
# Print ROC curve
plt.plot(fpr,tpr)
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
```

    AUC: 0.9465467667547905



```python
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_stepwise, pos_label=1)
# Print ROC curve
plt.plot(fpr,tpr)
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
```

    AUC: 0.9465467667547905
    


![png](output_47_1.png)



```python
###성능면에서는 네 모델이 큰 차이가 없음
```

# Lasso & RIdge


```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
```


```python

ploan_processed = ploan.dropna().drop(['ID','ZIP Code'], axis=1, inplace=False)

feature_columns = list(ploan_processed.columns.difference(["Personal Loan"]))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan'] # 대출여부: 1 or 0

train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y,train_size=0.7,test_size=0.3,random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
```

    (1750, 11) (750, 11) (1750,) (750,)
    


```python
ll =Lasso(alpha=0.01) ## lasso
ll.fit(train_x,train_y)

```




    Lasso(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
ll.coef_
```




    array([ 0.00000000e+00,  2.04783983e-03,  1.14390390e-01, -0.00000000e+00,
            6.58342418e-02,  4.76625359e-04,  3.13396711e-02,  3.55393865e-03,
            1.31719530e-05,  0.00000000e+00, -0.00000000e+00])




```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>Personal Loan</td>  <th>  No. Observations:  </th>   <td>  1750</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  1738</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>    11</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 23 Aug 2019</td> <th>  Pseudo R-squ.:     </th>   <td>0.6030</td>  
</tr>
<tr>
  <th>Time:</th>              <td>14:58:38</td>     <th>  Log-Likelihood:    </th>  <td> -229.35</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th>  <td> -577.63</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>2.927e-142</td>
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Age</th>                <td>    0.0245</td> <td>    0.102</td> <td>    0.240</td> <td> 0.810</td> <td>   -0.175</td> <td>    0.224</td>
</tr>
<tr>
  <th>CCAvg</th>              <td>    0.0985</td> <td>    0.063</td> <td>    1.562</td> <td> 0.118</td> <td>   -0.025</td> <td>    0.222</td>
</tr>
<tr>
  <th>CD Account</th>         <td>    4.3726</td> <td>    0.568</td> <td>    7.703</td> <td> 0.000</td> <td>    3.260</td> <td>    5.485</td>
</tr>
<tr>
  <th>CreditCard</th>         <td>   -1.2374</td> <td>    0.337</td> <td>   -3.667</td> <td> 0.000</td> <td>   -1.899</td> <td>   -0.576</td>
</tr>
<tr>
  <th>Education</th>          <td>    1.5203</td> <td>    0.190</td> <td>    7.999</td> <td> 0.000</td> <td>    1.148</td> <td>    1.893</td>
</tr>
<tr>
  <th>Experience</th>         <td>   -0.0070</td> <td>    0.102</td> <td>   -0.069</td> <td> 0.945</td> <td>   -0.206</td> <td>    0.192</td>
</tr>
<tr>
  <th>Family</th>             <td>    0.7579</td> <td>    0.128</td> <td>    5.914</td> <td> 0.000</td> <td>    0.507</td> <td>    1.009</td>
</tr>
<tr>
  <th>Income</th>             <td>    0.0547</td> <td>    0.004</td> <td>   12.659</td> <td> 0.000</td> <td>    0.046</td> <td>    0.063</td>
</tr>
<tr>
  <th>Mortgage</th>           <td>   -0.0001</td> <td>    0.001</td> <td>   -0.144</td> <td> 0.885</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>Online</th>             <td>   -0.4407</td> <td>    0.263</td> <td>   -1.674</td> <td> 0.094</td> <td>   -0.957</td> <td>    0.075</td>
</tr>
<tr>
  <th>Securities Account</th> <td>   -1.8520</td> <td>    0.561</td> <td>   -3.299</td> <td> 0.001</td> <td>   -2.952</td> <td>   -0.752</td>
</tr>
<tr>
  <th>const</th>              <td>  -13.9203</td> <td>    2.773</td> <td>   -5.021</td> <td> 0.000</td> <td>  -19.354</td> <td>   -8.486</td>
</tr>
</table>




```python
pred_y_lasso = ll.predict(test_x) # full model
pred_Y_lasso= cut_off(pred_y_lasso,0.5)
cfmat = confusion_matrix(test_y, pred_Y_lasso)
print(acc(cfmat))

```

    0.936
    


```python
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_lasso, pos_label=1)
# Print ROC curve
plt.plot(fpr,tpr)
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
```

    AUC: 0.9439995368672932
    


![png](output_56_1.png)



```python
rr =Ridge(alpha=0.01) ## lasso
rr.fit(train_x,train_y)

```




    Ridge(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)




```python
rr.coef_ ## ridge result
```




    array([-3.71283678e-03,  7.37570775e-03,  3.54973975e-01, -5.28579506e-02,
            7.83404224e-02,  4.12823466e-03,  3.62504712e-02,  3.27385112e-03,
            1.73105480e-06, -1.91297381e-02, -8.77388670e-02])




```python
ll.coef_ ## lasso result
```




    array([ 0.00000000e+00,  2.04783983e-03,  1.14390390e-01, -0.00000000e+00,
            6.58342418e-02,  4.76625359e-04,  3.13396711e-02,  3.55393865e-03,
            1.31719530e-05,  0.00000000e+00, -0.00000000e+00])




```python
pred_y_ridge = rr.predict(test_x) # full model
pred_Y_ridge= cut_off(pred_y_ridge,0.5)
cfmat = confusion_matrix(test_y, pred_Y_lasso)
print(acc(cfmat))

```

    0.936
    


```python
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_ridge, pos_label=1)
# Print ROC curve
plt.plot(fpr,tpr)
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
```

    AUC: 0.9494992377607533
    


![png](output_61_1.png)



```python
alpha = np.logspace(-3, 1, 5)
alpha
```




    array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01])




```python
data = []
acc_table=[]
for i, a in enumerate(alpha):
    lasso = Lasso(alpha=a).fit(train_x, train_y)
    data.append(pd.Series(np.hstack([lasso.intercept_, lasso.coef_])))
    pred_y = lasso.predict(test_x) # full model
    pred_y= cut_off(pred_y,0.5)
    cfmat = confusion_matrix(test_y, pred_y)
    acc_table.append((acc(cfmat)))
    

df_lasso = pd.DataFrame(data, index=alpha).T
df_lasso
acc_table_lasso = pd.DataFrame(acc_table, index=alpha).T
```


```python
df_lasso
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




```python
acc_table_lasso
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
      <th>0.001</th>
      <th>0.01</th>
      <th>0.1</th>
      <th>1.0</th>
      <th>10.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.932</td>
      <td>0.936</td>
      <td>0.894667</td>
      <td>0.897333</td>
      <td>0.897333</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = []
acc_table=[]
for i, a in enumerate(alpha):
    ridge = Ridge(alpha=a).fit(train_x, train_y)
    data.append(pd.Series(np.hstack([ridge.intercept_, ridge.coef_])))
    pred_y = ridge.predict(test_x) # full model
    pred_y= cut_off(pred_y,0.5)
    cfmat = confusion_matrix(test_y, pred_y)
    acc_table.append((acc(cfmat)))

    
df_ridge = pd.DataFrame(data, index=alpha).T
acc_table_ridge = pd.DataFrame(acc_table, index=alpha).T
```


```python
df_ridge
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
      <th>0.001</th>
      <th>0.01</th>
      <th>0.1</th>
      <th>1.0</th>
      <th>10.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.289557</td>
      <td>-0.289565</td>
      <td>-0.289645</td>
      <td>-0.290438</td>
      <td>-0.297581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.003713</td>
      <td>-0.003713</td>
      <td>-0.003713</td>
      <td>-0.003716</td>
      <td>-0.003723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.007376</td>
      <td>0.007376</td>
      <td>0.007376</td>
      <td>0.007378</td>
      <td>0.007388</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.355019</td>
      <td>0.354974</td>
      <td>0.354529</td>
      <td>0.350141</td>
      <td>0.311781</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.052866</td>
      <td>-0.052858</td>
      <td>-0.052782</td>
      <td>-0.052037</td>
      <td>-0.045541</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.078340</td>
      <td>0.078340</td>
      <td>0.078341</td>
      <td>0.078347</td>
      <td>0.078316</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.004128</td>
      <td>0.004128</td>
      <td>0.004129</td>
      <td>0.004136</td>
      <td>0.004175</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.036250</td>
      <td>0.036250</td>
      <td>0.036254</td>
      <td>0.036289</td>
      <td>0.036578</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.003274</td>
      <td>0.003274</td>
      <td>0.003274</td>
      <td>0.003278</td>
      <td>0.003313</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.019134</td>
      <td>-0.019130</td>
      <td>-0.019086</td>
      <td>-0.018655</td>
      <td>-0.014925</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.087756</td>
      <td>-0.087739</td>
      <td>-0.087569</td>
      <td>-0.085897</td>
      <td>-0.071545</td>
    </tr>
  </tbody>
</table>
</div>




```python
acc_table_ridge
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
      <th>0.001</th>
      <th>0.01</th>
      <th>0.1</th>
      <th>1.0</th>
      <th>10.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.932</td>
      <td>0.932</td>
      <td>0.932</td>
      <td>0.932</td>
      <td>0.932</td>
    </tr>
  </tbody>
</table>
</div>



## labmda값의 변화에 따른 회귀계수 축소 시각화


```python
import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
plt.semilogx(df_ridge.T)
plt.xticks(alpha)

ax2 = plt.subplot(122)
plt.semilogx(df_lasso.T)
plt.xticks(alpha)
plt.title("Lasso")

plt.show()
```

![image](https://user-images.githubusercontent.com/49333349/112686685-b6173400-8eb9-11eb-8a29-8c90a65f0335.png)

