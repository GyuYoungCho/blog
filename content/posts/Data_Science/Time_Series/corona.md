---
title: "Corona 확진자 분석 및 시계열 예측"
date: 2021-03-21T17:11:41+09:00
draft: false
categories : ["Data_Science", "Time_Series"]
---



[2-1-1](#2-1-1) : 코로나 인구대비 확진자수가 많은 상위 5개국 누적확진자수, 일일확진자수, 누적사망자수, 일일사망자수 선그래프로 시각화

[2-1-2](#2-1-2) : 코로나 검사자수, 확진자수, 완치자수, 사망자수, 인구수를 바탕으로 위험지수를 만들고 그 지수를 바탕으로 국가별 위험도를 판단, 상위 10개국에 대해 위험지수 막대그래프로 시각화

[2-1-3](#2-1-3) : 한국 누적 확진자수를 바탕으로 시계열 예측. 선형 시계열과 비선형 시계열 2가지로 모델링하고 평가. 5월 16일 이후 데이터로 테스트.



```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose 
```


```python
corona = pd.read_csv('../data/covid_19_data.csv',index_col = 'ObservationDate',parse_dates=True)
corona.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 109382 entries, 2020-01-22 to 2020-09-13
    Data columns (total 7 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   SNo             109382 non-null  int64  
     1   Province/State  75709 non-null   object 
     2   Country/Region  109382 non-null  object 
     3   Last Update     109382 non-null  object 
     4   Confirmed       109382 non-null  float64
     5   Deaths          109382 non-null  float64
     6   Recovered       109382 non-null  float64
    dtypes: float64(3), int64(1), object(3)
    memory usage: 6.7+ MB
    


```python
corona.head()
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
      <th>SNo</th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Last Update</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
    <tr>
      <th>ObservationDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-22</th>
      <td>1</td>
      <td>Anhui</td>
      <td>Mainland China</td>
      <td>1/22/2020 17:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-22</th>
      <td>2</td>
      <td>Beijing</td>
      <td>Mainland China</td>
      <td>1/22/2020 17:00</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-22</th>
      <td>3</td>
      <td>Chongqing</td>
      <td>Mainland China</td>
      <td>1/22/2020 17:00</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-22</th>
      <td>4</td>
      <td>Fujian</td>
      <td>Mainland China</td>
      <td>1/22/2020 17:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-22</th>
      <td>5</td>
      <td>Gansu</td>
      <td>Mainland China</td>
      <td>1/22/2020 17:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# <span style="color:blue">2-1-1</span>


```python
corona.index.unique()
```

    DatetimeIndex(['2020-01-22', '2020-01-23', '2020-01-24', '2020-01-25',
                   '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29',
                   '2020-01-30', '2020-01-31',
                   ...
                   '2020-09-04', '2020-09-05', '2020-09-06', '2020-09-07',
                   '2020-09-08', '2020-09-09', '2020-09-10', '2020-09-11',
                   '2020-09-12', '2020-09-13'],
                  dtype='datetime64[ns]', name='ObservationDate', length=236, freq=None)




```python
last_day = corona.loc[corona.index == '2020-09-13']
last_day_country = last_day.groupby('Country/Region')[['Confirmed','Deaths','Recovered']].sum()
last_day_country
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>38716.0</td>
      <td>1420.0</td>
      <td>31638.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>11353.0</td>
      <td>334.0</td>
      <td>6569.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>48254.0</td>
      <td>1612.0</td>
      <td>34037.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>1344.0</td>
      <td>53.0</td>
      <td>943.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>3388.0</td>
      <td>134.0</td>
      <td>1301.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>30574.0</td>
      <td>221.0</td>
      <td>20082.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>2011.0</td>
      <td>583.0</td>
      <td>1212.0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>13539.0</td>
      <td>312.0</td>
      <td>12260.0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>7526.0</td>
      <td>224.0</td>
      <td>5678.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
Confirmed_list = last_day_country.sort_values(by='Confirmed',ascending=False)[:5].index
Deaths_list = last_day_country.sort_values(by='Deaths',ascending=False)[:5].index
last_day_country.sort_values(by='Confirmed',ascending=False)
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>6519573.0</td>
      <td>194071.0</td>
      <td>2451406.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>4754356.0</td>
      <td>78586.0</td>
      <td>3702595.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>4330455.0</td>
      <td>131625.0</td>
      <td>3723206.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>1059024.0</td>
      <td>18517.0</td>
      <td>873684.0</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>722832.0</td>
      <td>30526.0</td>
      <td>559321.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>23.0</td>
      <td>0.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>17.0</td>
      <td>0.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>12.0</td>
      <td>0.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>MS Zaandam</th>
      <td>9.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
Confirmed_list
```

    Index(['US', 'India', 'Brazil', 'Russia', 'Peru'], dtype='object', name='Country/Region')




```python
Top_Confrimed = corona.loc[corona['Country/Region'].isin(Confirmed_list)]
Top_Deaths = corona.loc[corona['Country/Region'].isin(Deaths_list)]
```

```python
five_top_Confirmed = Top_Confrimed.groupby(['Country/Region','ObservationDate'])['Confirmed'].sum()
five_top_Deaths = Top_Deaths.groupby(['Country/Region','ObservationDate'])['Deaths'].sum()
```

```python
five_top_Confirmed.unstack('Country/Region').plot()
five_top_Deaths.unstack('Country/Region').plot()
```


![image](https://user-images.githubusercontent.com/49333349/111909407-8377d080-8aa0-11eb-8155-19f2fc0c98cb.png)
![image](https://user-images.githubusercontent.com/49333349/111909426-94284680-8aa0-11eb-815b-53c53aec1b2b.png)


```python
f_u = five_top_Confirmed.unstack('Country/Region')
f_d = five_top_Deaths.unstack('Country/Region')
one_day_Confirmed = f_u - f_u.shift(1)
one_day_Deaths = f_d - f_d.shift(1)
```


```python
one_day_Confirmed 
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
      <th>Country/Region</th>
      <th>Brazil</th>
      <th>India</th>
      <th>Peru</th>
      <th>Russia</th>
      <th>US</th>
    </tr>
    <tr>
      <th>ObservationDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-22</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-23</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-01-25</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-26</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-09</th>
      <td>35816.0</td>
      <td>95735.0</td>
      <td>4615.0</td>
      <td>5172.0</td>
      <td>34256.0</td>
    </tr>
    <tr>
      <th>2020-09-10</th>
      <td>40557.0</td>
      <td>96551.0</td>
      <td>6586.0</td>
      <td>5310.0</td>
      <td>35286.0</td>
    </tr>
    <tr>
      <th>2020-09-11</th>
      <td>43718.0</td>
      <td>97570.0</td>
      <td>7291.0</td>
      <td>5421.0</td>
      <td>47192.0</td>
    </tr>
    <tr>
      <th>2020-09-12</th>
      <td>33523.0</td>
      <td>94372.0</td>
      <td>6603.0</td>
      <td>5406.0</td>
      <td>41471.0</td>
    </tr>
    <tr>
      <th>2020-09-13</th>
      <td>14768.0</td>
      <td>0.0</td>
      <td>6162.0</td>
      <td>5361.0</td>
      <td>34359.0</td>
    </tr>
  </tbody>
</table>
</div>


```python
one_day_Confirmed.fillna(0).plot()
one_day_Deaths.fillna(0).plot()
plt.ylim(-100,5000)
```

    (-100.0, 5000.0)

![image](https://user-images.githubusercontent.com/49333349/111909437-a2766280-8aa0-11eb-9896-4c0dd25c3e50.png)
![image](https://user-images.githubusercontent.com/49333349/111909446-adc98e00-8aa0-11eb-8e2b-6d097d73fa14.png)

# <span style="color:blue">2-1-2</span>

- risk = (확진자 수 - 사망자 수 - 완치자 수)/(state * 10000)
- 국가별 1달 risk 계산 평균 -> top 10


```python
one_month = corona['2020-08-14':]
a = one_month.groupby(['ObservationDate','Country/Region'])[['Confirmed','Deaths','Recovered']].sum()
a
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
    <tr>
      <th>ObservationDate</th>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2020-08-14</th>
      <th>Afghanistan</th>
      <td>37431.0</td>
      <td>1363.0</td>
      <td>26714.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>7117.0</td>
      <td>219.0</td>
      <td>3695.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>37664.0</td>
      <td>1351.0</td>
      <td>26308.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>989.0</td>
      <td>53.0</td>
      <td>863.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>1852.0</td>
      <td>86.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2020-09-13</th>
      <th>West Bank and Gaza</th>
      <td>30574.0</td>
      <td>221.0</td>
      <td>20082.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>2011.0</td>
      <td>583.0</td>
      <td>1212.0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>13539.0</td>
      <td>312.0</td>
      <td>12260.0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>7526.0</td>
      <td>224.0</td>
      <td>5678.0</td>
    </tr>
  </tbody>
</table>

</div>




```python
a.reset_index(level=0)
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
      <th>ObservationDate</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>2020-08-14</td>
      <td>37431.0</td>
      <td>1363.0</td>
      <td>26714.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>2020-08-14</td>
      <td>7117.0</td>
      <td>219.0</td>
      <td>3695.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>2020-08-14</td>
      <td>37664.0</td>
      <td>1351.0</td>
      <td>26308.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>2020-08-14</td>
      <td>989.0</td>
      <td>53.0</td>
      <td>863.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>2020-08-14</td>
      <td>1852.0</td>
      <td>86.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>2020-09-13</td>
      <td>30574.0</td>
      <td>221.0</td>
      <td>20082.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>2020-09-13</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>2020-09-13</td>
      <td>2011.0</td>
      <td>583.0</td>
      <td>1212.0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>2020-09-13</td>
      <td>13539.0</td>
      <td>312.0</td>
      <td>12260.0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>2020-09-13</td>
      <td>7526.0</td>
      <td>224.0</td>
      <td>5678.0</td>
    </tr>
  </tbody>
</table>

</div>


```python
b = last_day.groupby('Country/Region')['SNo'].count()
b.name = 'State_Num'
b.sort_values(ascending=False)[:10]
```

    Country/Region
    Russia            83
    US                58
    Japan             49
    India             37
    Colombia          33
    Mexico            32
    Mainland China    31
    Ukraine           27
    Brazil            27
    Peru              26
    Name: State_Num, dtype: int64




```python
df = a.reset_index(level=0).join(b, how='left')
```

```python
df.set_index('ObservationDate',append=True,inplace=True)
```

```python
df
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>State_Num</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th>ObservationDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Afghanistan</th>
      <th>2020-08-14</th>
      <td>37431.0</td>
      <td>1363.0</td>
      <td>26714.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-08-15</th>
      <td>37551.0</td>
      <td>1370.0</td>
      <td>27166.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-08-16</th>
      <td>37596.0</td>
      <td>1375.0</td>
      <td>27166.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-08-17</th>
      <td>37599.0</td>
      <td>1375.0</td>
      <td>27166.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-08-18</th>
      <td>37599.0</td>
      <td>1375.0</td>
      <td>27166.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Zimbabwe</th>
      <th>2020-09-09</th>
      <td>7429.0</td>
      <td>222.0</td>
      <td>5542.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-09-10</th>
      <td>7453.0</td>
      <td>222.0</td>
      <td>5635.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-09-11</th>
      <td>7479.0</td>
      <td>224.0</td>
      <td>5660.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-09-12</th>
      <td>7508.0</td>
      <td>224.0</td>
      <td>5675.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-09-13</th>
      <td>7526.0</td>
      <td>224.0</td>
      <td>5678.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
df['risk_pi'] = (df['Confirmed'] - df['Deaths'] - df['Recovered'])/(df['State_Num']*10000)
df.head()
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>State_Num</th>
      <th>risk_pi</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th>ObservationDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Afghanistan</th>
      <th>2020-08-14</th>
      <td>37431.0</td>
      <td>1363.0</td>
      <td>26714.0</td>
      <td>1</td>
      <td>0.9354</td>
    </tr>
    <tr>
      <th>2020-08-15</th>
      <td>37551.0</td>
      <td>1370.0</td>
      <td>27166.0</td>
      <td>1</td>
      <td>0.9015</td>
    </tr>
    <tr>
      <th>2020-08-16</th>
      <td>37596.0</td>
      <td>1375.0</td>
      <td>27166.0</td>
      <td>1</td>
      <td>0.9055</td>
    </tr>
    <tr>
      <th>2020-08-17</th>
      <td>37599.0</td>
      <td>1375.0</td>
      <td>27166.0</td>
      <td>1</td>
      <td>0.9058</td>
    </tr>
    <tr>
      <th>2020-08-18</th>
      <td>37599.0</td>
      <td>1375.0</td>
      <td>27166.0</td>
      <td>1</td>
      <td>0.9058</td>
    </tr>
  </tbody>
</table>
</div>




```python
# last_day_country['risk_pi'] = (last_day_country['Confirmed']/1000000 - last_day_country['Recovered']/1000000)/1000000
# top_risk_list = last_day_country.sort_values(by='risk_pi',ascending=False)[:10]
df_risk = df.groupby('Country/Region')['risk_pi'].mean()
top_risk_list = df_risk.sort_values(ascending=False)[:10]
top_risk_list
```




    Country/Region
    Bangladesh      10.533694
    Argentina       10.164006
    South Africa     7.527106
    US               6.239020
    Philippines      6.162177
    Belgium          5.627332
    Iraq             5.155116
    Bolivia          5.124435
    Honduras         4.621997
    Romania          4.466106
    Name: risk_pi, dtype: float64




```python
top_risk_list.plot.bar()
```




    <AxesSubplot:xlabel='Country/Region'>




![image](https://user-images.githubusercontent.com/49333349/111909459-bd48d700-8aa0-11eb-8fdc-c11fc9620642.png)


# <span style="color:blue">2-1-3</span>


```python
k_c = corona[corona['Country/Region']=='South Korea']
print(len(k_c))
print(len(k_c.index.unique()))
```

    236
    236
    


```python
from statsmodels.tsa.stattools import adfuller  # 정상성 판별 여부
```


```python
k = k_c['Confirmed']
korea_co = (k - k.shift(1)).dropna()
korea_co.plot()
```




    <AxesSubplot:xlabel='ObservationDate'>




![image](https://user-images.githubusercontent.com/49333349/111909468-c9cd2f80-8aa0-11eb-8526-46ba1f680a6e.png)




```python
train = korea_co[:'2020-05-15']
test = korea_co['2020-05-16':'2020-06-15']
```


```python
adfuller(train)
```




    (-2.692880412737049,
     0.07527839224613403,
     5,
     108,
     {'1%': -3.4924012594942333,
      '5%': -2.8886968193364835,
      '10%': -2.5812552709190673},
     1147.051801657943)



`1차 차분


```python
diff1 = (train - train.shift(1)).dropna()
adfuller(diff1)
```




    (-5.228238034317768,
     7.697004182819284e-06,
     2,
     110,
     {'1%': -3.4912451337340342,
      '5%': -2.8881954545454547,
      '10%': -2.5809876033057852},
     1139.877102390829)




```python
decomposition = seasonal_decompose(diff1)
decomposition.plot()
plt.show()
```


![image](https://user-images.githubusercontent.com/49333349/111909484-d94c7880-8aa0-11eb-828f-c9b2c6feab70.png)




```python
diff2 = (diff1 - diff1.shift(1)).dropna()
adfuller(diff2)
```




    (-6.291128356594684,
     3.5970963612700875e-08,
     6,
     105,
     {'1%': -3.4942202045135513,
      '5%': -2.889485291005291,
      '10%': -2.5816762131519275},
     1140.9946617995597)



2차 차분


```python
decomposition = seasonal_decompose(diff2)
decomposition.plot()
plt.show()
```


![image](https://user-images.githubusercontent.com/49333349/111909506-ed907580-8aa0-11eb-875f-6c1a69e2e712.png)




```python
plot_acf(diff1);
plot_pacf(diff1);
```


![image](https://user-images.githubusercontent.com/49333349/111909520-fa14ce00-8aa0-11eb-8c79-5bf7239206b7.png)




![image](https://user-images.githubusercontent.com/49333349/111909533-0567f980-8aa1-11eb-93b1-bdd686ca4bdc.png)




```python
model = sm.tsa.statespace.SARIMAX(train,
        order=[1,1,1],trend='t')
```

    C:\Users\Gyu\Anaconda3\envs\py37\lib\site-packages\statsmodels\tsa\base\tsa_model.py:165: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      % freq, ValueWarning)
    


```python
result = model.fit()
result.summary()
```




<table class="simpletable">
<caption>Statespace Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Confirmed</td>    <th>  No. Observations:  </th>    <td>114</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 1, 1)</td> <th>  Log Likelihood     </th> <td>-641.888</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 27 Sep 2020</td> <th>  AIC                </th> <td>1291.777</td>
</tr>
<tr>
  <th>Time:</th>                <td>10:40:43</td>     <th>  BIC                </th> <td>1302.687</td>
</tr>
<tr>
  <th>Sample:</th>             <td>01-23-2020</td>    <th>  HQIC               </th> <td>1296.204</td>
</tr>
<tr>
  <th></th>                   <td>- 05-15-2020</td>   <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>drift</th>  <td>   -0.0210</td> <td>    0.162</td> <td>   -0.129</td> <td> 0.897</td> <td>   -0.339</td> <td>    0.297</td>
</tr>
<tr>
  <th>ar.L1</th>  <td>   -0.1768</td> <td>    0.215</td> <td>   -0.821</td> <td> 0.412</td> <td>   -0.599</td> <td>    0.245</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.0869</td> <td>    0.199</td> <td>   -0.437</td> <td> 0.662</td> <td>   -0.476</td> <td>    0.302</td>
</tr>
<tr>
  <th>sigma2</th> <td> 5025.8920</td> <td>  309.739</td> <td>   16.226</td> <td> 0.000</td> <td> 4418.815</td> <td> 5632.969</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>34.23</td> <th>  Jarque-Bera (JB):  </th> <td>350.68</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.73</td>  <th>  Prob(JB):          </th>  <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.01</td>  <th>  Skew:              </th>  <td>-0.03</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>  <td>11.63</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
pred = result.predict(start='2020-05-16',end='2020-06-15')
```


```python
train.plot(label='Train')
test.plot(label='Test')
pred.plot(label='pred')
```




    <AxesSubplot:xlabel='ObservationDate'>




![image](https://user-images.githubusercontent.com/49333349/111909544-1153bb80-8aa1-11eb-819e-4eac4abf08f5.png)




```python
from sklearn.metrics import mean_squared_error
mean_squared_error(test,pred)
```




    3631.3045190567364




```python
model = sm.tsa.statespace.SARIMAX(train,
        order=[1,1,1], trend = [0,1,1])
result = model.fit()
result.summary()
```

    C:\Users\Gyu\Anaconda3\envs\py37\lib\site-packages\statsmodels\tsa\base\tsa_model.py:165: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      % freq, ValueWarning)
    




<table class="simpletable">
<caption>Statespace Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Confirmed</td>    <th>  No. Observations:  </th>    <td>114</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 1, 1)</td> <th>  Log Likelihood     </th> <td>-641.925</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 27 Sep 2020</td> <th>  AIC                </th> <td>1293.851</td>
</tr>
<tr>
  <th>Time:</th>                <td>10:42:26</td>     <th>  BIC                </th> <td>1307.488</td>
</tr>
<tr>
  <th>Sample:</th>             <td>01-23-2020</td>    <th>  HQIC               </th> <td>1299.384</td>
</tr>
<tr>
  <th></th>                   <td>- 05-15-2020</td>   <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>drift</th>   <td>   -0.0008</td> <td>    0.735</td> <td>   -0.001</td> <td> 0.999</td> <td>   -1.441</td> <td>    1.439</td>
</tr>
<tr>
  <th>trend.2</th> <td>   -0.0002</td> <td>    0.016</td> <td>   -0.015</td> <td> 0.988</td> <td>   -0.031</td> <td>    0.031</td>
</tr>
<tr>
  <th>ar.L1</th>   <td>   -0.1758</td> <td>    0.224</td> <td>   -0.785</td> <td> 0.433</td> <td>   -0.615</td> <td>    0.263</td>
</tr>
<tr>
  <th>ma.L1</th>   <td>   -0.0878</td> <td>    0.207</td> <td>   -0.425</td> <td> 0.671</td> <td>   -0.493</td> <td>    0.317</td>
</tr>
<tr>
  <th>sigma2</th>  <td> 5216.6180</td> <td>  347.385</td> <td>   15.017</td> <td> 0.000</td> <td> 4535.756</td> <td> 5897.480</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>34.24</td> <th>  Jarque-Bera (JB):  </th> <td>350.89</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.73</td>  <th>  Prob(JB):          </th>  <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.01</td>  <th>  Skew:              </th>  <td>-0.04</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>  <td>11.63</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
pred = result.predict(start='2020-05-16',end='2020-06-15')
train.plot(label='Train')
test.plot(label='Test')
pred.plot(label='pred')
```




    <AxesSubplot:xlabel='ObservationDate'>




![image](https://user-images.githubusercontent.com/49333349/111909560-26304f00-8aa1-11eb-8314-c5a9b3f26567.png)



```python
mean_squared_error(test,pred)
```




    6478.176590606611




```python
from statsmodels.tsa.api import ExponentialSmoothing
model = ExponentialSmoothing(np.asarray(train),
                trend='add')
model_result = model.fit()
```


```python
model_result.summary()
```




<table class="simpletable">
<caption>ExponentialSmoothing Model Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>endog</td>        <th>  No. Observations:  </th>        <td>114</td>      
</tr>
<tr>
  <th>Model:</th>            <td>ExponentialSmoothing</td> <th>  SSE                </th>    <td>571027.819</td>   
</tr>
<tr>
  <th>Optimized:</th>                <td>True</td>         <th>  AIC                </th>      <td>979.165</td>    
</tr>
<tr>
  <th>Trend:</th>                  <td>Additive</td>       <th>  BIC                </th>      <td>990.110</td>    
</tr>
<tr>
  <th>Seasonal:</th>                 <td>None</td>         <th>  AICC               </th>      <td>979.950</td>    
</tr>
<tr>
  <th>Seasonal Periods:</th>         <td>None</td>         <th>  Date:              </th> <td>Sun, 27 Sep 2020</td>
</tr>
<tr>
  <th>Box-Cox:</th>                  <td>False</td>        <th>  Time:              </th>     <td>10:46:34</td>    
</tr>
<tr>
  <th>Box-Cox Coeff.:</th>           <td>None</td>         <th>                     </th>         <td> </td>       
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>                 <th>coeff</th>                <th>code</th>               <th>optimized</th>     
</tr>
<tr>
  <th>smoothing_level</th> <td>           0.7647937</td> <td>               alpha</td> <td>                True</td>
</tr>
<tr>
  <th>smoothing_slope</th> <td>            0.000000</td> <td>                beta</td> <td>                True</td>
</tr>
<tr>
  <th>initial_level</th>   <td>            0.000000</td> <td>                 l.0</td> <td>                True</td>
</tr>
<tr>
  <th>initial_slope</th>   <td>           0.1845664</td> <td>                 b.0</td> <td>                True</td>
</tr>
</table>




```python
y_hat = pd.DataFrame(test.copy())
y_hat['ES'] = model_result.forecast(len(test))
mean_squared_error(y_hat['Confirmed'],y_hat['ES'])
```




    360.9693621017652




```python
train.plot(label='Train')
test.plot(label='Test')
y_hat['ES'].plot(label='ES')
plt.legend()
```

![image](https://user-images.githubusercontent.com/49333349/111909897-70660000-8aa2-11eb-950c-77cb76d0d556.png)

```python
model = ExponentialSmoothing(np.asarray(train+1),trend='mul')
model_result = model.fit()
model_result.summary()
```

<table class="simpletable">
<caption>ExponentialSmoothing Model Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>endog</td>        <th>  No. Observations:  </th>        <td>114</td>      
</tr>
<tr>
  <th>Model:</th>            <td>ExponentialSmoothing</td> <th>  SSE                </th>    <td>559193.862</td>   
</tr>
<tr>
  <th>Optimized:</th>                <td>True</td>         <th>  AIC                </th>      <td>976.778</td>    
</tr>
<tr>
  <th>Trend:</th>               <td>Multiplicative</td>    <th>  BIC                </th>      <td>987.723</td>    
</tr>
<tr>
  <th>Seasonal:</th>                 <td>None</td>         <th>  AICC               </th>      <td>977.563</td>    
</tr>
<tr>
  <th>Seasonal Periods:</th>         <td>None</td>         <th>  Date:              </th> <td>Sun, 27 Sep 2020</td>
</tr>
<tr>
  <th>Box-Cox:</th>                  <td>False</td>        <th>  Time:              </th>     <td>10:47:32</td>    
</tr>
<tr>
  <th>Box-Cox Coeff.:</th>           <td>None</td>         <th>                     </th>         <td> </td>       
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>                 <th>coeff</th>                <th>code</th>               <th>optimized</th>     
</tr>
<tr>
  <th>smoothing_level</th> <td>           0.7749651</td> <td>               alpha</td> <td>                True</td>
</tr>
<tr>
  <th>smoothing_slope</th> <td>            0.000000</td> <td>                beta</td> <td>                True</td>
</tr>
<tr>
  <th>initial_level</th>   <td>           0.4787151</td> <td>                 l.0</td> <td>                True</td>
</tr>
<tr>
  <th>initial_slope</th>   <td>           0.9574234</td> <td>                 b.0</td> <td>                True</td>
</tr>
</table>

```python
y_hat = pd.DataFrame(test.copy())
y_hat['ES'] = model_result.forecast(len(test))
mean_squared_error(y_hat['Confirmed'],y_hat['ES'])
```


    995.7063961726025




```python
train.plot(label='Train')
test.plot(label='Test')
y_hat['ES'].plot(label='ES')
plt.legend()
```


    <matplotlib.legend.Legend at 0x2212c6de470>


![image](https://user-images.githubusercontent.com/49333349/111909917-8378d000-8aa2-11eb-8d4e-e0ed491ff5d3.png)



