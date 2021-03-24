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

` id_vars로 id를 지정할 수 있는데 칼럼으로 들어감

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


---

**재구조화 : pivot_table**

pd.pivot_table(data, index, columns, values, aggfunc)


```python
import numpy as np
import pandas as pd

data = pd.DataFrame({'cust_id': ['c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c3', 'c3', 'c3'],
'prod_cd': ['p1', 'p2', 'p3', 'p1', 'p2', 'p3', 'p1', 'p2', 'p3'],
'grade' : ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
 'pch_amt': [30, 10, 0, 40, 15, 30, 0, 0, 10]})
data
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
      <th>cust_id</th>
      <th>prod_cd</th>
      <th>grade</th>
      <th>pch_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c1</td>
      <td>p1</td>
      <td>A</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c1</td>
      <td>p2</td>
      <td>A</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c1</td>
      <td>p3</td>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c2</td>
      <td>p1</td>
      <td>A</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c2</td>
      <td>p2</td>
      <td>A</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c2</td>
      <td>p3</td>
      <td>A</td>
      <td>30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c3</td>
      <td>p1</td>
      <td>B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>c3</td>
      <td>p2</td>
      <td>B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c3</td>
      <td>p3</td>
      <td>B</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.pivot(index='cust_id', columns='prod_cd', values='pch_amt')
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
      <th>prod_cd</th>
      <th>p1</th>
      <th>p2</th>
      <th>p3</th>
    </tr>
    <tr>
      <th>cust_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c1</th>
      <td>30</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>c2</th>
      <td>40</td>
      <td>15</td>
      <td>30</td>
    </tr>
    <tr>
      <th>c3</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


- aggfuc : 중복되는 값이 여러 개 있을 경우 집계할 수 있는 함수 제공

```python
pd.pivot_table(data, index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.mean)
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
      <th>prod_cd</th>
      <th>p1</th>
      <th>p2</th>
      <th>p3</th>
      <th>All</th>
    </tr>
    <tr>
      <th>grade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>35.000000</td>
      <td>12.500000</td>
      <td>15.000000</td>
      <td>20.833333</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>All</th>
      <td>23.333333</td>
      <td>8.333333</td>
      <td>13.333333</td>
      <td>15.000000</td>
    </tr>
  </tbody>
</table>
</div>


- margins=True : 모든 행과 열 기준으로 집계

```python
pd.pivot_table(data, index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.sum, margins=True)
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
      <th>prod_cd</th>
      <th>p1</th>
      <th>p2</th>
      <th>p3</th>
      <th>All</th>
    </tr>
    <tr>
      <th>grade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>70</td>
      <td>25</td>
      <td>30</td>
      <td>125</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>All</th>
      <td>70</td>
      <td>25</td>
      <td>40</td>
      <td>135</td>
    </tr>
  </tbody>
</table>
</div>


---
<br>

**stack, unstack**

- stack : 위에서 아래로 쌓는 것
- unstack 쌓은 것을 왼쪽에서 오른쪽으로 늘어놓는 것


```python
mul_index = pd.MultiIndex.from_tuples([('cust_1', '2015'), ('cust_1', '2016'),('cust_2', '2015'), ('cust_2', '2016')])
data = pd.DataFrame(data=np.arange(16).reshape(4, 4),
        index=mul_index,columns=['prd_1', 'prd_2', 'prd_3', 'prd_4'], dtype='int')
data
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
      <th></th>
      <th>prd_1</th>
      <th>prd_2</th>
      <th>prd_3</th>
      <th>prd_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">cust_1</th>
      <th>2015</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">cust_2</th>
      <th>2015</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.stack(level = -1, dropna=False)
```


    cust_1  2015  prd_1     0
                  prd_2     1
                  prd_3     2
                  prd_4     3
            2016  prd_1     4
                  prd_2     5
                  prd_3     6
                  prd_4     7
    cust_2  2015  prd_1     8
                  prd_2     9
                  prd_3    10
                  prd_4    11
            2016  prd_1    12
                  prd_2    13
                  prd_3    14
                  prd_4    15
    dtype: int32


- level은 default로 -1로 상위에 있는 것을 올리고 내림.

```python
data.unstack(level=-1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">prd_1</th>
      <th colspan="2" halign="left">prd_2</th>
      <th colspan="2" halign="left">prd_3</th>
      <th colspan="2" halign="left">prd_4</th>
    </tr>
    <tr>
      <th></th>
      <th>2015</th>
      <th>2016</th>
      <th>2015</th>
      <th>2016</th>
      <th>2015</th>
      <th>2016</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cust_1</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>cust_2</th>
      <td>8</td>
      <td>12</td>
      <td>9</td>
      <td>13</td>
      <td>10</td>
      <td>14</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>


- level이 0 이상 숫자 : 왼쪽에서 숫자만큼의 index 칼럼을 올림

```python
data.unstack(level=0)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">prd_1</th>
      <th colspan="2" halign="left">prd_2</th>
      <th colspan="2" halign="left">prd_3</th>
      <th colspan="2" halign="left">prd_4</th>
    </tr>
    <tr>
      <th></th>
      <th>cust_1</th>
      <th>cust_2</th>
      <th>cust_1</th>
      <th>cust_2</th>
      <th>cust_1</th>
      <th>cust_2</th>
      <th>cust_1</th>
      <th>cust_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015</th>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>4</td>
      <td>12</td>
      <td>5</td>
      <td>13</td>
      <td>6</td>
      <td>14</td>
      <td>7</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



 
```python
data.unstack(level=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">prd_1</th>
      <th colspan="2" halign="left">prd_2</th>
      <th colspan="2" halign="left">prd_3</th>
      <th colspan="2" halign="left">prd_4</th>
    </tr>
    <tr>
      <th></th>
      <th>2015</th>
      <th>2016</th>
      <th>2015</th>
      <th>2016</th>
      <th>2015</th>
      <th>2016</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cust_1</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>cust_2</th>
      <td>8</td>
      <td>12</td>
      <td>9</td>
      <td>13</td>
      <td>10</td>
      <td>14</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




**multi index 교차표**

pd.crosstab([id1, id2], [col1, col2])
