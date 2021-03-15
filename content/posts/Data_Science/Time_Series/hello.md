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


비정상 시계열이란 위의 4가지 패턴이 들어가 있는 경우를 말하고 이 요소들을 제거하고 정상화하는 것이 필수적이다.

일반적인 시계열 분석
1. 시각화
2. 시계열 안정
3. 자기상관성 및 parameter 찾기
4. ARIMA 모델 등으로 예측


참고 : [자주 쓰는 numpy/pandas](https://s3.amazonaws.com/quandl-static-content/Documents/Quandl+-+Pandas,+SciPy,+NumPy+Cheat+Sheet.pdf)
