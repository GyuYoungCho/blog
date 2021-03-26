---
title: "pandas 사용"
date: 2021-03-27T03:11:41+09:00
draft: false
categories : ["Data_Science", "Data_Handling"]
---

**map**

Series만
```python
df['winning_rate']  = df['team'].map(lambda x : total_record(x)[3])
```

**apply**

복수 개의 컬럼
```python
df['winning_rate'] = df.apply(lambda x:relative_record(x['team'], x['against'])[3], axis=1)
```
