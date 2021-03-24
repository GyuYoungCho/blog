---
title: "연관 분석"
date: 2021-03-24T00:11:41+09:00
draft: false
categories : ["Data_Science", "Association_Rule"]
---

## 연관 분석
` 거래 또는 사건들 간의 규칙을 발견하여 IF-THEN 구조로 결과의 연관성을 파악

#### 연관 규칙 측도
- 지지도(support) : 전체 거래 중 A와 B를 동시에 포함하는 비율
	- $P(A \cap B) $
- 신뢰도(confidence) : A 거래중 A와 B를 동시에 포함하는 비율
	- $P(B | A)$
- 향상도(lift) : A가 구매되지 않았을 때 B의 구매확률에 비해 A가 구매되었을 때 B의 구매확률의 증가비
	- $P(B | A) / P(B)$

### Apriori
최소 지지도 이상의 빈발항목집합을 찾은 후 연관규칙 계산

---
<br>

## apply

회차와 6개 숫자 정보가 있는 로또 데이터를 이용

```python
import numpy as np
import pandas as pd
lotto = pd.read_csv('data/lotto.csv')
lotto.time_id = lotto['time_id'].astype('str')
lotto.head()
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
      <th>time_id</th>
      <th>num1</th>
      <th>num2</th>
      <th>num3</th>
      <th>num4</th>
      <th>num5</th>
      <th>num6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>859</td>
      <td>8</td>
      <td>22</td>
      <td>35</td>
      <td>38</td>
      <td>39</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>858</td>
      <td>9</td>
      <td>13</td>
      <td>32</td>
      <td>38</td>
      <td>39</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>857</td>
      <td>6</td>
      <td>10</td>
      <td>16</td>
      <td>28</td>
      <td>34</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>856</td>
      <td>10</td>
      <td>24</td>
      <td>40</td>
      <td>41</td>
      <td>43</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>855</td>
      <td>8</td>
      <td>15</td>
      <td>17</td>
      <td>19</td>
      <td>43</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pd.melt(lotto, id_vars='time_id')
lotto_ary = lotto.set_index('time_id').T.to_dict('list')
```

#### Transaction data로 변환하기

각 아이템이 있는지 없는지 보여줌

```python
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(lotto_ary.values()).transform(lotto_ary.values())
df = pd.DataFrame(te_ary, columns=te.columns_)
df.head()
```

![image](https://user-images.githubusercontent.com/49333349/112262468-0bccc000-8cb1-11eb-8240-80524fccfd63.png)


```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
fre_set = apriori(df, min_support=0.002, use_colnames=True) # 최소 지지도 0.002
# fre_set['length'] = fre_set['itemsets'].apply(lambda x: len(x))

fre_rule = association_rules(fre_set,metric="confidence", min_threshold=0.8)
# 최소 신뢰도 0.8
```


```python
fre_rule.sort_values(by="lift",ascending=False).iloc[:10,:]
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>703</th>
      <td>(16, 26, 31)</td>
      <td>(43, 36)</td>
      <td>0.002328</td>
      <td>0.012806</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>78.090909</td>
      <td>0.002298</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>643</th>
      <td>(24, 34, 22)</td>
      <td>(31, 7)</td>
      <td>0.002328</td>
      <td>0.012806</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>78.090909</td>
      <td>0.002298</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>642</th>
      <td>(34, 31, 7)</td>
      <td>(24, 22)</td>
      <td>0.002328</td>
      <td>0.012806</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>78.090909</td>
      <td>0.002298</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>682</th>
      <td>(26, 21, 14)</td>
      <td>(18, 15)</td>
      <td>0.002328</td>
      <td>0.013970</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>71.583333</td>
      <td>0.002296</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>652</th>
      <td>(34, 10, 36)</td>
      <td>(44, 22)</td>
      <td>0.002328</td>
      <td>0.013970</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>71.583333</td>
      <td>0.002296</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>646</th>
      <td>(24, 22, 31)</td>
      <td>(34, 7)</td>
      <td>0.002328</td>
      <td>0.013970</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>71.583333</td>
      <td>0.002296</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>666</th>
      <td>(24, 20, 15)</td>
      <td>(12, 30)</td>
      <td>0.002328</td>
      <td>0.013970</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>71.583333</td>
      <td>0.002296</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>702</th>
      <td>(16, 26, 43)</td>
      <td>(36, 31)</td>
      <td>0.002328</td>
      <td>0.013970</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>71.583333</td>
      <td>0.002296</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>700</th>
      <td>(16, 43, 36)</td>
      <td>(26, 31)</td>
      <td>0.002328</td>
      <td>0.015134</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>66.076923</td>
      <td>0.002293</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>653</th>
      <td>(34, 10, 22)</td>
      <td>(36, 44)</td>
      <td>0.002328</td>
      <td>0.016298</td>
      <td>0.002328</td>
      <td>1.0</td>
      <td>61.357143</td>
      <td>0.002290</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>

<br>

---

번호 개수를 시각화보면 다음과 같다

```python
k = []
data = np.array([k+_ for _ in lotto_ary.values()]).flatten()
data = pd.Series(data)
import seaborn as sns
sns.barplot(x=data.value_counts(ascending=False).index[:10] ,y = data.value_counts(ascending=False)[:10], order=data.value_counts(ascending=False).index[:10])
```

![image](https://user-images.githubusercontent.com/49333349/112319155-2030ad00-8cf1-11eb-9acb-08b7436f9711.png)

34가 제일 많이 나와서 34가 나오는 규칙을 추출해 보았다.

```python
fre_rule[conse.astype('str').str.contains('34')]
```

![image](https://user-images.githubusercontent.com/49333349/112319510-7ef62680-8cf1-11eb-957f-e54d5efb34d9.png)

여기서 향상도는 6.4정도로 나오는데 1번 규칙만 살펴보면 34만 추출된 것 보다 1,5,13이 뽑히고 34가 뽑힐 확률이 6.4배 정도 된다는 뜻이다.
하지만 순서를 고려하지 않고 단순히 조합에 대한 확률만 고려한 규칙이라 여기서는 향상도가 높은 조합이 추첨번호가 될 가능성이 높은 것은 아님.

그 외에도 인사이트를 찾아보기 위해 describe() 등 해보는 게 좋음.

데이터 출처 : 데이터 에듀 adp 모의고사