---
title: "가설 검정"
date: 2021-03-23T23:11:41+09:00
draft: false
categories : ["Data_Science", "Statistics"]
---


### 1. 통계적 가설 검정
모집단의 특성에 대한 가설에 대한 통계적 유의성 검정
	- 통계적 유의성 -> 확률적으로 봐서 단순한 우연이 아님

과정

1. 귀무 가설 대립 가설 설정
2. 검정 통계량 설정
3. 기각역 설정
4. 검정통계량 계산
5. 의사 결정

**가설 검정 오류**
제 1종 오류 : 참인데 거짓이라 함
제 2종 오류 : 거짓인데 참

p-value : 귀무 가설이 맞다고 가정할 때 얻은 결과와 다른 결과가 관측될 확률, 귀무 가설을 기각할 근거가 됨.

---

```python
from scipy.stats import *
```

###  T 검정

```python
# 단일 표본 검정
ttest_1samp(ser_1, popmean = 5)
# pvalue가 대략적으로 0.05 미만이면기각
```

    Ttest_1sampResult(statistic=-2.499999999999999, pvalue=0.04652823228416732)

```python
# 대응 표본 검정
ttest_rel(ser_1, ser_2)
```

    Ttest_relResult(statistic=-1.219715097075045, pvalue=0.2683379268893624)

```python
# 독립 표본 검정(등분산 X)
ttest_ind(ser_1, ser_2, equal_var = False)
```

    Ttest_indResult(statistic=-1.684024198163435, pvalue=0.11958234592838302)

#### 독립성 검정(카이제곱 검정)

```python
df_chi2 = pd.DataFrame({"ID": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                        "YN": ["Y", "N", "N", "Y", "N", "Y", "Y", "N", "Y", "Y"]})

df_crosstab = pd.crosstab(df_chi2["ID"], df_chi2["YN"])
chi2_contingency(df_crosstab)
# 통계량, P-VALUE, DF, 기대도수
```

    (0.41666666666666663,
     0.5186050164287255,
     1,
     array([[2., 3.],
            [2., 3.]]))

```python
df_crosstab 
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
      <th>YN</th>
      <th>N</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

### 정규성

정규성을 만족하는지 확인하는 방법은 shapiro, kstest, qqplot 등이 있다.
```python

ser_nor = pd.Series([4, 2, 5, 6, 7, 4, 5, 2, 6, 1, 3, 0, 15])
from scipy.stats import shapiro
shapiro(ser_nor)
# 통계량, p-value, 0.05이상이어야 정규성을 만족한다 할 수 있음.
```
    (0.8451933264732361, 0.02470559999346733)
정규성 검정은 샘플 개수에 대해 민감한 편이므로 여러 방면으로 확인해 보는 것이 좋다.


#### 등분산성
정규성을 만족하는 경우와 아닌 경우

```python
# 표본이 정규성

ser_1 = pd.Series([2, 5, 3, 4, 6, 2, 3])
ser_2 = pd.Series([7, 3, 6, 5, 2, 6, 7])

bartlett(ser_1, ser_2)
```

    BartlettResult(statistic=0.3574441170696212, pvalue=0.549929145265077)


```python
# 표본이 정규 X
levene(ser_1, ser_2)
```

    LeveneResult(statistic=0.1666666666666668, pvalue=0.6902818588864357)

---
<br>

## 일원분산분석(Anova)
셋 이상의 그룹에서 차이가 존재하는지 확인

```python
ser_1 = pd.Series([2, 5, 3, 4, 6, 2, 3])
ser_2 = pd.Series([7, 3, 6, 5, 2, 6, 7])
ser_3 = pd.Series([9, 11, 4, 8, 2, 15, 3])

df_aov = pd.DataFrame([ser_1, ser_2, ser_3]).transpose().melt()

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

formula = "value ~ C(variable)" # C() 범주형 변수임을 명시
lm = ols(formula, df_aov).fit()
anova_lm(lm)
# p-value 0.05이하여야 유의미한 차이가 있다고 할 수 있음.
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
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(variable)</th>
      <td>2.0</td>
      <td>52.666667</td>
      <td>26.333333</td>
      <td>2.783557</td>
      <td>0.088449</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>18.0</td>
      <td>170.285714</td>
      <td>9.460317</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

#### 사후 검정

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(df_aov["value"], df_aov["variable"]))
# reject가 true면 차이 있음
```

    Multiple Comparison of Means - Tukey HSD, FWER=0.05
    ===================================================
    group1 group2 meandiff p-adj   lower  upper  reject
    ---------------------------------------------------
         0      1   1.5714 0.6082 -2.6229 5.7658  False
         0      2   3.8571 0.0746 -0.3372 8.0515  False
         1      2   2.2857 0.3675 -1.9087 6.4801  False
    ---------------------------------------------------
    

---
<br>

## 상관분석
연속형 변수 간에 선형 관계 파악

```python
df = pd.DataFrame([ser_1, ser_2, ser_3]).T
pd.plotting.scatter_matrix(df);
```

![image](https://user-images.githubusercontent.com/49333349/112162121-1a729300-8c2f-11eb-8b89-24435e22c05e.png)

```python
print(pearsonr(df.iloc[:,0], df.iloc[:,1]))
print(pearsonr(df.iloc[:,1], df.iloc[:,2]))
print(pearsonr(df.iloc[:,0], df.iloc[:,2]))
# 상관계수, p-value, 0.05미만이면 유의한 상관성이 있다고 봄
```

    (-0.9359709753334592, 0.0019245719846704063)
    (0.13695501944225102, 0.7696722723761817)
    (-0.4136643973298085, 0.3562507831530036)
    
- 스피어만 : 두 변수 순위의 단조 관련성을 측정. 변수의 분포가 심각하게 정규성을 벗어나거나 순위형 자료일 때 사용

```python
# 스피어만 상관계수를 dataframe으로 보기
df.corr(method = 'spearman')
```
![image](https://user-images.githubusercontent.com/49333349/112162411-602f5b80-8c2f-11eb-928a-6635002be11c.png)

**비선형 관계를 보려면 시각화 해보는 게 좋음**