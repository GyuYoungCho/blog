---
title: "시계열 데이터 이해"
date: 2021-03-15T17:11:41+09:00
draft: false
categories : ["Data_Science", "Time_Series"]
---

# 시계열 데이터

1. 시계열 데이터 요소

- 추세(Trend): 장기적으로 나타나는 변동 패턴
- 계절성(Seasonal): 주,월,분기,반기 단위 등 이미 알려진 시간의 주기로 나타나는 패턴
- 주기(Cyclic): 고정된 기간이 아닌 장기적인 변동
- 랜덤요소 (random/residual/remainder)


`비정상 시계열이란 위의 4가지 패턴이 들어가 있는 경우를 말하고 이 요소들을 제거하고 정상화하는 것이 필수적이다.

**y_t = Level + Trend + Seasonality + Noise**

```python
from statsmodels.tsa.seasonal import seasonal_decompose
series = pd.Series([i + np.random.randint(10) for i in range(1,100)])
result = seasonal_decompose(series, freq=1)
result.plot();
```

![image](https://user-images.githubusercontent.com/49333349/111330823-a1fe5600-86b3-11eb-8bcd-5a660859c735.png)


### Stationary(정상성)
평균,분산 공분산이 시간에 따라 변하지 않아야 함.

#### 일반적인 시계열 분석
1. 시각화
2. 시계열 안정
3. 자기상관성 및 parameter 찾기
4. ARIMA 모델 등으로 예측

### 안정성을 위한 과정
#### Differencing
- 앞 데이터와 뒤 데이터의 차이
- stationary하게 만드는 과정 -> d parameter
- trend, seasonality가 없어지는 과정 -> 보통 1,2차에서 끝나고 그래도 성능이 좋아지지 않으면 다른 요인의 문제

```python
# stationary 확인
from statsmodels.tsa.stattools import adfuller

def adf_check(ts):
    result = adfuller(ts)
    if result[1] <= 0.05:
        print('Stationary {}'.format(result[1]))
    else:
        print('Non-Stationary {}'.format(result[1]))

# 보통 0.05이하가 되어야 
adf_check(timeseries) 

# 1차 차분
diff1 = timeseries - timeseries.shift(1)
adf_check(diff1.dropna())
```

#### 로그 변환
- 표준편차가 자료의 크기에 비례하여 증가할 경우

### 정상과정 확률 모형
가우시안 백색잡음의 현재값과 과거값들의 선형조합으로 이루어져 있다고 가정

#### Autoregression(AR)
- p parameter 관련
- 데이터와 이전 시점 사이의 관계에 대한 회귀 모델

#### Moving Average(MA)
- q parameter
- 이전 시점의 moveing average의 residual에 대한 회귀 모델 -> noise 예측

#### ARMA (Auto-Regressive Moving Average)
- AR모형과 MA모형의 특징을 모두 가지는 모형

### ARIMA(p,d,q)
- 


4 5
c.xc
....
xx..
...x
x..x


참고  : 작년 코로나 데이터 분석한 것을 복습 + 추가 개념 정리
[Corona data anaylsis](https://github.com/GyuYoungCho/ADP_STUDY/blob/master/study/time_series.ipynb)
[자주 쓰는 numpy/pandas](https://s3.amazonaws.com/quandl-static-content/Documents/Quandl+-+Pandas,+SciPy,+NumPy+Cheat+Sheet.pdf)
(https://h3imdallr.github.io/2017-08-19/arima/)
