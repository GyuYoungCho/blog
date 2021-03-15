---
title: "Java_Programming1"
date: 2021-03-15T16:12:07+09:00
draft: false
categories : ["Programming", "Python"]
---

# Python Basic

### lambda

함수를 간단하게 표현. 일회성을 지님

`lambda 인자 : 표현식`

```python
(lambda x,y: x + y)(10, 20)
# 30
```

---

### map

리스트로부터 원소를 하나씩 꺼내서 함수를 적용하고 새로운 리스트에 담음

`map(함수, 리스트)`

```python
list(map(lambda x: x ** 2, range(5)))
# [0, 1, 4, 9, 16]
```

---

### reduce

순서형 자료(문자열, 리스트, 튜플 )의 원소를 순서대로 함수에 적용

`reduce(함수, sequence)`

```python
reduce(lambda x, y: y + x, 'abcde')
# 'edcba'
```

x를 기존 문자, y를 새로운 문자로 생각하면 새로운 원소가 기존 원소의 앞에 붙는다고 생각하면 위와 같이 역순으로 나온다.

---

### filter

리스트로부터 원소를 함수에 적용시키고 결과가 참인 값들로 새 리스트를 만듬

`filter(함수, 리스트)`

```python
list(filter(lambda x: x < 5, range(10)))
# [0, 1, 2, 3, 4]
list(filter(lambda x: x % 2, range(10)))
# [1, 3, 5, 7, 9]
```