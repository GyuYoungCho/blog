---
title: "SQL"
date: 2021-04-11T23:12:07+09:00
draft: false
categories : ["Programming", "SQL"]
---

`

## DDL

DDL: 데이터 형태 규정: create, alter, drop, rename, truncate

unsigned : 0이상 양수

auto increment : 자동 1 증가

check : 값 범위

## TCL

TCL: commit, rollback, savepoint

```sql
# db 생성
create database testdb;
# db사용
use testdb

```

**ALTER 시 고려사항**

20자 입력할 수 있는 데이터

크기를 늘리면 상관없지만 줄이게 되면 데이터 훼손이 일어날 수 있음.

## Transaction

데이터 처리의 한 단위( 개별 SQL 중심이 아닌 업무 중심)

Transaction 시작과 종료

- TX의 시작 : DML(C,U,D)를 시작하면 시작
- TX의 종료 : 명시적으로 commit(반영) 또는 rollback(취소)이 호출될 때
    - DDL(crate, alter, drop 등 구조 변경)이 실행될 때 --> commit
    - SQL 툴의 정상적인 종료 --> commit, 비정상 종료 --> rollback

TX가 종료(commit, rollback)되지 않았는데 다른 사용자가 조회 등을 하려고 하면  rock이 걸리게 되어 사용할 수 없는 상황이 발생하게 됨.

---

## DML

INSERT INTO TABLE() VALUES ()

UPDATE TABLE SET ~ WHERE CONDITION

DELETE FROM TABLE WHERE CONDITION

## SELECT

BETWEEN  : 이상 이하

LIKE %X% : X포함

%X__ : 뒤에서 3번째가 X

### Limit 구문

limit 4, 5 : 앞의 네 개는 스킵하고 5번째부터

### case ~ when 구문

```sql
select deptno,
		case when job='ALAYST' then sal end 'ALAYST',
		case when job='CLERK' then sal fend 'CLERK',
		case when job='MANAGER' then sal end 'MANAGER',
		case when job='PRESIDENT' then sal end 'PRESIDENT',
		case when job='SALESMAN' then sal end 'SALESMAN'
from emp;
```

## 숫자 함수

FLOOR(값) : 값보다 작은 정수 중 가장 큰 수 (실수를 무조건 버림)

TRUNCATE(값, 자릿수) : 소수점 이하 버리기

ROUND(값, 자릿수) : 반올림

## 문자 함수

SUBSTRING('문자열', 시작 위치, 개수) : 문자열 중 시작 위치부터 개수만큼 출력

SUBSTR도 가능

CONCAT('문자열1', '문자열2', '문자열3' ...) : 문자열 잇기

UPPER('문자열') : 대문자로

REVERSE('문자열') : 문자열을 반대로 나열

LEFT('문자열', 개수) : 왼쪽에서 개수만큼

REPLACE(문자,바뀔거,새거)

INSERT(문자, 시작위치, 길이, 새문자)

TRIM('문자열')

LENGTH()는 바이트 수, char_length()는 글자 수

INSTR(문자, 찾는문자) : 위치값 리턴

## 날짜 함수

now() : 날짜, 시간

sysdate() : 날짜, 시간

curdate() : 날짜

curtime() : 시간

DATE_ADD, SUB

DAYOFWEEK : 일요일 1 토요일 7

WEEKDAY : 월요일 0 일요일 6

sleep해서 재우면 sysdate()는 10초가 늘어남

NULL은 논리연산시 NULL|TRUE = TRUE, NULL&FALSE = FALSE 나머지 NULL

---

## JOIN

**OUTER JOIN**

한 쪽에는 데이터가 존재하는데 다른 쪽 테이블에는 없을 경우 검색되지 않는 문제 해결

## Sub query

단일 행

다중 행 → in, any, all

=any → in

---

**SQL 실행 순서**

```sql
SELECT 5
FROM 1
WHERE 2
GROUP BY 3
HAVING 4
ORDER BY 6
```

Aggregation : sum, max, count 등

- 집계함수를 사용하지 않은 필드랑 같이 쓸 수 없음 → group by 사용

Having : group by 한 결과에 조건 추가(집계 조건)

SET 연산

- select 절의 column의 개수, type이 일치해야 한다

union : 중복x 

union all : 중복

intersect : 교집합

minus : 차집합

---

## Database Modeling

### Primary Key

- 후보키 중에서 선택한 주 키, not null, unique

alternate key : 후보키 중 기본 키 아님

composite : 둘 이상 컬럼을 식별자로

### Foreign Key

- 관계를 맺는 두 엔티티에서 서로 참조하는 릴레이션의 애트리뷰트로 지정되는 키 값

관계가 만족되지 않는 경우 optional(동그라미)

### Mapping Rule

단순 엔티티 ⇒ 테이블

속성 ⇒ 컬럼

식별자 ⇒ 기본키

관계 ⇒ 참조키. 테이블

## 정규화

정규화가 제대로 이루어지지 않는다면 수정, 삽입, 삭제 과정에서 문제가 발생할 수 있음

- 하나를 수정할 때 여러 개를 수정, 삭제할 때 여러 개가 삭제

### 제1정규화

- 중복을 제거 : 각 row마다 컬럼의 값이 1개씩만 있어야 함
- 반복되는 그룹속성 : 같은 성격과 내용의 컬럼이 연속적으로 나타나는 컬럼

### 제2정규화

- 복합키에 전첵적으로 의존하지 않는 속성 제거
- 부분적 함수 종속 관계 : 기본키 중에 특정 칼럼에만 종속된 칼럼이 없어야 함
    - Student, Subject가 복합키고 age가 student에만 종속이면 (Student, Subject), (Student, age)로 나누기

### 제3정규화

- 이행적 함수 종속 제거 : 기본키 이외의 다른 컬럼이 그외 다른 컬럼을 결정할 수 없음
- 기본키에 의존하지 않고 일반 컬럼에 의존하는 칼럼 제거(기본키 이외 속성이 다른 칼럼을 제어)

### 역정규화 방법

- 데이터 중복 : 조인 프로세스를 줄이기 위함
- 테이블 분리 : 컬럼 기준으로 분리(컬럼 수), 레코드 기준으로 분리(레코드 양)
- 요약 테이블 생성 : 조회 프로세스를 줄이기 위해 정보만을 저장하는 테이블을 생성
- 테이블 통합 : 분리된 두 테이블이 시스템 성능에 영향을 끼칠 경우 고려