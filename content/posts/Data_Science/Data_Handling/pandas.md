---
title: "pandas 사용"
date: 2021-03-23T00:11:41+09:00
draft: false
categories : ["Data_Science", "Data_Handling"]
---

**melt**

```python
ser_1 = pd.Series([2, 5, 3, 4, 6, 2, 3])
ser_2 = pd.Series([7, 3, 6, 5, 2, 6, 7])
ser_3 = pd.Series([9, 11, 4, 8, 2, 15, 3])

df_aov = pd.DataFrame([ser_1, ser_2, ser_3]).transpose().melt()
df_aov
```

![image](https://user-images.githubusercontent.com/49333349/112160209-4b51c880-8c2d-11eb-9b97-c13cccd9ff58.png)


**cut**
값을 기반으로 이산화
```python
pd.cut([5, 20, 29, 33, 41], bins = 5, right = False)
```

    [[5.0, 12.2), [19.4, 26.6), [26.6, 33.8), [26.6, 33.8), [33.8, 41.036)]
    Categories (5, interval[float64]): [[5.0, 12.2) < [12.2, 19.4) < [19.4, 26.6) < [26.6, 33.8) < [33.8, 41.036)]

**qcut**
특정 분위수를 계산해 이산화
```python
pd.qcut([5, 20, 29, 33, 41], [0,0.2,0.4,0.6,0.8,1])
```

    [(4.999, 17.0], (17.0, 25.4], (25.4, 30.6], (30.6, 34.6], (34.6, 41.0]]
    Categories (5, interval[float64]): [(4.999, 17.0] < (17.0, 25.4] < (25.4, 30.6] < (30.6, 34.6] < (34.6, 41.0]]