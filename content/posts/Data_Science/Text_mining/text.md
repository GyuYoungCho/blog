---
title: "Text mining"
date: 2021-03-26T20:50:41+09:00
draft: false
categories : ["Data_Science", "Text_mining"]
---

## 자연어 처리


konply : 한국어 처리
scikit-learn - feature_extraction.text : 문서 전처리 


#### apply
# review 데이터와 사전 데이터 가져오기
```python
k = []
with open('data/영화 기생충_review.txt','r') as f:
    for _ in f.readlines():
        k.append(_.split('\n')[0])

d= []
with open('data/영화 기생충_사전.txt','r') as g:
    for _ in g.readlines():
        d.append(_.split('\n')[0])
    
data = pd.Series(k)
data.head()
```




    0                             별1개  준 사람들은   나베당임
    1                                             역쉬
    2         영화가 끝나고 가슴이 먹먹하고 답답햇습니다 너무나 충격적이었습니다..
    3    지금까지 나온 감독의 모든 작품이 압축되어있다는 느낌을 받음.  Bomb!!!
    4                           대단한 영화. 몰입력 장난아님. 후아
    dtype: object




```python
from string import punctuation
punctuation
```

    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
import re
data = data.str.lower() # 소문자
data = data.str.replace(r'[0-9]','')  # 숫자 제거
data = data.str.replace('[' + punctuation + ']','') # 구두점 제거
data = data.str.strip()
data.head()
```


    0                          별개  준 사람들은   나베당임
    1                                         역쉬
    2       영화가 끝나고 가슴이 먹먹하고 답답햇습니다 너무나 충격적이었습니다
    3    지금까지 나온 감독의 모든 작품이 압축되어있다는 느낌을 받음  bomb
    4                         대단한 영화 몰입력 장난아님 후아
    dtype: object


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
tdm = cv.fit_transform(data)
```


```python
from konlpy.tag import Okt
ma = Okt().nouns(' '.join([_ for _ in data]))
ok = Okt().pos(' '.join([_ for _ in data]))
```


```python
# 동사, 명사, 형용사 뽑기
list = []
for du in ok:
    if du[1] in ['Noun','Verb','Adjective']:
        list.append(du[0])
        
list
```

series로 바꾸고 value_counts로 개수 세기



   
```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ['why hello there', 'omg hello pony', 'she went there? omg']
vec = CountVectorizer()
X = vec.fit_transform(data)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 
df1 = df.sum(axis=0)



```python
df2 = df1[df1.index.isin(d)]
df2
```




    기생충    17
    기정      1
    박사장     1
    박소담     2
    봉준호    78
    송강호    29
    이선균    10
    이정은     3
    장혜진     1
    조여정    10
    최우식     4
    dtype: int64




```python
'|'.join([_ for _ in d])
```

    '기생충봉준호송강호기택이선균박사장조여정연교최우식기우박소담기정장혜진충숙이정은이지혜박서준'



```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='white').generate(' '.join([_ for _ in data]))
plt.imshow(wordcloud, interpolation='bilinear')
```
![image](https://user-images.githubusercontent.com/49333349/112681657-5f5a2c00-8eb2-11eb-9147-a1ad89d6a99d.png)


```python
from wordcloud import STOPWORDS
wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='white').generate_from_frequencies(df2.to_dict())
plt.imshow(wordcloud, interpolation='bilinear')
# 외국어면 stopwords = STOPWORDS 설정, countvector에서도 설정 가능
```

---

tfidf

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do',
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
```
	[[0.         0.46735098 0.         0.46735098 0.         0.46735098
  	0.         0.35543247 0.46735098]
 	[0.         0.         0.79596054 0.         0.         0.
  	0.         0.60534851 0.        ]
 	[0.57735027 0.         0.         0.         0.57735027 0.
  	0.57735027 0.         0.        ]]
	{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}



---

영어

```python
word_tokens = nltk.word_tokenize(cleaned_content)
tokens_pos = nltk.pos_tag(word_tokens)

NN_words = []
for word, pos in tokens_pos:
    if 'NN' in pos:
        NN_words.append(word)

# 형용사 RB, 동사 VB

for word in unique_NN_words:
    if word in stopwords_list:
        while word in final_NN_words: final_NN_words.remove(word)

# 빈도

from collections import Counter
c = Counter(final_NN_words) # input type should be a list of words (or tokens)
print(c)
k = 20
print(c.most_common(k)) # 빈도수 기준 상위 k개 단어 출력
```