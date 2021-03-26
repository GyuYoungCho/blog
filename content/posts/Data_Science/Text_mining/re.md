---
title: "정규 표현식"
date: 2021-03-26T19:00:41+09:00
draft: false
categories : ["Data_Science", "Text_mining"]
---



### 1. 정규표현식(re) 에 대한 이해 및 숙지

* 정규표현식 
 - regular expression
 - 특정한 패턴과 일치하는 문자열를 '검색', '치환', '제거' 하는 기능을 지원
 - 정규표현식의 도움없이 패턴을 찾는 작업(Rule 기반)은 불완전 하거나, 작업의 cost가 높음
 - e.g) 이메일 형식 판별, 전화번호 형식 판별, 숫자로만 이루어진 문자열 등

* **raw string**
 - 문자열 앞에 r이 붙으면 해당 문자열이 구성된 그대로 문자열로 변환


`기본 패턴
- a, X, 9 등등 문자 하나하나의 character들은 정확히 해당 문자와 일치
  - e.g) 패턴 test는 test 문자열과 일치
  - 대소문자의 경우 기본적으로 구별하나, 구별하지 않도록 설정 가능
- 몇몇 문자들에 대해서는 예외가 존재하는데, 이들은 틀별한 의미로 사용 됨
  - . ^ $ * + ? { } [ ] \ | ( )
 
 - . (마침표) - 어떤 한개의 character와 일치 (newline(엔터) 제외)
 
 - \w - 문자 character와 일치 [a-zA-Z0-9_]
 - \s - 공백문자와 일치
 - \t, \n, \r - tab, newline, return
 - \d - 숫자 character와 일치 [0-9]
 - ^ = 시작, $ = 끝 각각 문자열의 시작과 끝을 의미
 - \가 붙으면 스페셜한 의미가 없어짐. 예를들어 \\.는 .자체를 의미 \\\는 \를 의미
 - 자세한 내용은 링크 참조 https://docs.python.org/3/library/re.html

#### **search method**
 - 첫번째로 패턴을 찾으면 match 객체를 반환
 - 패턴을 찾지 못하면 None 반환


```python
import re
```


```python
m = re.search(r'abc', '123abdef')
m
```


```python
m = re.search(r'\d\d\d\w', '112abcdef119')
m
```




    <re.Match object; span=(0, 4), match='112a'>




```python
m = re.search(r'..\w\w', '@#$%ABCDabcd')
m
```




    <re.Match object; span=(2, 6), match='$%AB'>



#### **metacharacters (메타 캐릭터)**

#### **[]** 문자들의 범위를 나타내기 위해 사용
   - [] 내부의 메타 캐릭터는 캐릭터 자체를 나타냄
   - e.g)
   - [abck] : a or b or c or k
   - [abc.^] : a or b or c or . or ^
   - [a-d]  : -와 함께 사용되면 해당 문자 사이의 범위에 속하는 문자 중 하나
   - [0-9]  : 모든 숫자
   - [a-z]  : 모든 소문자
   - [A-Z]  : 모든 대문자
   - [a-zA-Z0-9] : 모든 알파벳 문자 및 숫자
   - [^0-9] : ^가 맨 앞에 사용 되는 경우 해당 문자 패턴이 아닌 것과 매칭


```python
re.search(r'[cbm]at', 'aat')
```


```python
re.search(r'[0-4]haha', '7hahah')
```


```python
re.search(r'[abc.^]aron', 'daron')
```


```python
re.search(r'[^abc]aron', '0aron')
```




    <re.Match object; span=(0, 5), match='0aron'>



#### \ 
 1. 다른 문자와 함께 사용되어 특수한 의미를 지님
   - \d : 숫자를          [0-9]와 동일
   - \D : 숫자가 아닌 문자  [^0-9]와 동일
   - \s : 공백 문자(띄어쓰기, 탭, 엔터 등)
   - \S : 공백이 아닌 문자
   - \w : 알파벳대소문자, 숫자 [0-9a-zA-Z]와 동일
   - \W : non alpha-numeric 문자 [^0-9a-zA-Z]와 동일
 2. 메타 캐릭터가 캐릭터 자체를 표현하도록 할 경우 사용
   - \\. , \\\



```python
re.search(r'\Sand', 'apple land banana')
```




    <re.Match object; span=(6, 10), match='land'>




```python
re.search(r'\.and', '.and')
```




    <re.Match object; span=(0, 4), match='.and'>



#### **.** 
 - 모든 문자를 의미


```python
re.search(r'p.g', 'pig')
```




    <re.Match object; span=(0, 3), match='pig'>



#### **반복패턴**
 - 패턴 뒤에 위치하는 *, +, ?는 해당 패턴이 반복적으로 존재하는지 검사 
   - '+' -> 1번 이상의 패턴이 발생
   - '*' -> 0번 이상의 패턴이 발생
   - '?' -> 0 혹은 1번의 패턴이 발생
 - 반복을 패턴의 경우 greedy하게 검색 함, 즉 가능한 많은 부분이 매칭되도록 함
  - e.g) a[bcd]*b  패턴을 abcbdccb에서 검색하는 경우
    - ab, abcb, abcbdccb 전부 가능 하지만 최대한 많은 부분이 매칭된 abcbdccb가 검색된 패턴


```python
re.search(r'a[bcd]*b', 'abcbdccb')
```




    <re.Match object; span=(0, 8), match='abcbdccb'>




```python
re.search(r'b\w+a', 'banana')
```




    <re.Match object; span=(0, 6), match='banana'>




```python
re.search(r'i+', 'piigiii')
```




    <re.Match object; span=(1, 3), match='ii'>




```python
re.search(r'pi+g', 'pg')
```


```python
re.search(r'pi*g', 'pg')
```




    <re.Match object; span=(0, 2), match='pg'>




```python
re.search(r'https?', 'http://www.naver.com')
```




    <re.Match object; span=(0, 4), match='http'>



#### **^**, **$**
 - ^  문자열의 맨 앞부터 일치하는 경우 검색
 - \$  문자열의 맨 뒤부터 일치하는 경우 검색


```python
re.search(r'b\w+a', 'cabana')
```




    <re.Match object; span=(2, 6), match='bana'>




```python
re.search(r'^b\w+a', 'cabana')
```


```python
re.search(r'^b\w+a', 'babana')
```




    <re.Match object; span=(0, 6), match='babana'>




```python
re.search(r'b\w+a$', 'cabana')
```




    <re.Match object; span=(2, 6), match='bana'>




```python
re.search(r'b\w+a$', 'cabanap')
```

 #### **grouping**
  - ()을 사용하여 그루핑
  - 매칭 결과를 각 그룹별로 분리 가능
  - 패턴 명시 할 때, 각 그룹을 괄호() 안에 넣어 분리하여 사용


```python
m = re.search(r'(\w+)@(.+)', 'test@gmail.com')
print(m.group(1))
print(m.group(2))
print(m.group(0))
```

    test
    gmail.com
    test@gmail.com
    

 #### **{}**
  - *, +, ?을 사용하여 반복적인 패턴을 찾는 것이 가능하나, 반복의 횟수 제한은 불가
  - 패턴뒤에 위치하는 중괄호{}에 숫자를 명시하면 해당 숫자 만큼의 반복인 경우에만 매칭
  - {4} - 4번 반복
  - {3,4} - 3 ~ 4번 반복


```python
re.search('pi{3,5}g', 'piiiiig')
```




    <re.Match object; span=(0, 7), match='piiiiig'>



#### **미니멈 매칭(non-greedy way)**
 - 기본적으로 *, +, ?를 사용하면 greedy(맥시멈 매칭)하게 동작함
 - *?, +?을 이용하여 해당 기능을 구현


```python
re.search(r'<.+>', '<html>haha</html>')
```




    <re.Match object; span=(0, 17), match='<html>haha</html>'>




```python
re.search(r'<.+?>', '<html>haha</html>')
```




    <re.Match object; span=(0, 6), match='<html>'>



#### **{}?**
 - {m,n}의 경우 m번 에서 n번 반복하나 greedy하게 동작
 - {m,n}?로 사용하면 non-greedy하게 동작. 즉, 최소 m번만 매칭하면 만족


```python
re.search(r'a{3,5}', 'aaaaa')
```




    <re.Match object; span=(0, 5), match='aaaaa'>




```python
re.search(r'a{3,5}?', 'aaaaa')
```




    <re.Match object; span=(0, 3), match='aaa'>



#### **match**
 - search와 유사하나, 주어진 문자열의 시작부터 비교하여 패턴이 있는지 확인
 - 시작부터 해당 패턴이 존재하지 않다면 None 반환


```python
re.match(r'\d\d\d', 'my number is 123')
```


```python
re.match(r'\d\d\d', '123 is my number')
```




    <re.Match object; span=(0, 3), match='123'>




```python
re.search(r'^\d\d\d', '123 is my number')
```




    <re.Match object; span=(0, 3), match='123'>



#### **findall**
 - search가 최초로 매칭되는 패턴만 반환한다면, findall은 매칭되는 전체의 패턴을 반환
 - 매칭되는 모든 결과를 리스트 형태로 반환


```python
re.findall(r'[\w-]+@[\w.]+', 'test@gmail.com haha test2@gmail.com nice test test')
```




    ['test@gmail.com', 'test2@gmail.com']



#### **sub**
 - 주어진 문자열에서 일치하는 모든 패턴을 replace
 - 그 결과를 문자열로 다시 반환함
 - 두번째 인자는 특정 문자열이 될 수도 있고, 함수가 될 수 도 있음
 - count가 0인 경우는 전체를, 1이상이면 해당 숫자만큼 치환 됨


```python
re.sub(r'[\w-]+@[\w.]+', 'great', 'test@gmail.com haha test2@gmail.com nice test test', count=1)
```




    'great haha test2@gmail.com nice test test'



#### **compile**
 - 동일한 정규표현식을 매번 다시 쓰기 번거로움을 해결
 - compile로 해당표현식을 re.RegexObject 객체로 저장하여 사용가능


```python
email_reg = re.compile(r'[\w-]+@[\w.]+')
email_reg.search('test@gmail.com haha good')
# email_reg.findall()
```




    <re.Match object; span=(0, 14), match='test@gmail.com'>



### 연습문제 
  - 아래 뉴스에서 이메일 주소를 추출해 보세요
  - 다음중 올바른 (http, https) 웹페이지만 찾으시오


```python
import requests
from bs4 import BeautifulSoup
# 위의 두 모듈이 없는 경우에는 pip install requests bs4 실행

def get_news_content(url):
    response = requests.get(url)
    content = response.text

    soup = BeautifulSoup(content, 'html5lib')

    div = soup.find('div', attrs = {'id' : 'harmonyContainer'})
    
    content = ''
    for paragraph in div.find_all('p'):
        content += paragraph.get_text()
        
    return content

news1 = get_news_content('https://news.v.daum.net/v/20190617073049838')
print(news1)

```

    (로스앤젤레스=연합뉴스) 옥철 특파원 = 팀 쿡 애플 최고경영자(CEO)가 16일(현지시간) 실리콘밸리 앞마당 격인 미국 서부 명문 스탠퍼드대학 학위수여식에서 테크기업들을 향해 쓴소리를 쏟아냈다.쿡은 이날 연설에서 실리콘밸리 테크기업들은 자신들이 만든 혼란에 대한 책임을 질 필요가 있다고 경고했다.근래 IT 업계의 가장 큰 이슈인 개인정보 침해, 사생활 보호 문제를 콕 집어 라이벌인 구글, 페이스북 등 IT 공룡을 겨냥한 발언이라는 해석이 나왔다.쿡은 "최근 실리콘밸리 산업은 고귀한 혁신과는 점점 더 거리가 멀어지는 것으로 알려져 있다. 책임을 받아들이지 않고도 신뢰를 얻을 수 있다는 그런 믿음 말이다"라고 꼬집었다.개인정보 유출 사건으로 미 의회 청문회에 줄줄이 불려 나간 경쟁사 CEO들을 향해 일침을 가한 것으로 보인다.그는 또 실리콘밸리에서 희대의 사기극을 연출한 바이오벤처 스타트업 테라노스(Theranos)를 직격했다.쿡은 "피 한 방울로 거짓된 기적을 만들 수 있다고 믿었느냐"면서 "이런 식으로 혼돈의 공장을 만든다면 그 책임에서 절대 벗어날 수 없다"라고 비난했다.테라노스는 손가락 끝을 찔러 극미량의 혈액 샘플만 있으면 각종 의학정보 분석은 물론 거의 모든 질병 진단이 가능한 바이오헬스 기술을 개발했다고 속여 월가 큰손들로부터 거액의 투자를 유치했다가 해당 기술이 사기인 것으로 드러나 청산한 기업이다.쿡은 애플의 경우 프라이버시(사생활) 보호에 초점을 맞춘 새로운 제품 기능들로 경쟁사들에 맞서고 있다며 자사의 데이터 보호 정책을 은근히 홍보하기도 했다.oakchul@yna.co.kr저작권자(c)연합뉴스. 무단전재-재배포금지
    


```python
email_reg = re.compile(r'[\w-]+@[\w.]+\w+')
email_reg.search(news1)
```


    <re.Match object; span=(774, 795), match='oakchul@yna.co.kr저작권자'>


```python
webs = ['http://www.test.co.kr', 
        'https://www.test1.com', 
        'http://www.test.com', 
        'ftp://www.test.com', 
        'http:://www.test.com',
       'htp://www.test.com',
       'http://www.google.com', 
       'https://www.homepage.com.']

```

```python
web_reg = re.compile(r'https?://[\w.]+\w+$')
list(map(lambda w:web_reg.search(w) != None, webs))
```

    [True, True, True, False, False, False, True, False]


