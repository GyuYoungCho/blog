---
title: "Java_Programming1"
date: 2021-01-19T16:12:07+09:00
draft: false
categories : ["Programming", "Java"]
---

# Java Basic

**환경**

- jdk : 소프트웨어 개발
- jre : 실행 환경

#### 다음 파일(HelloWorld.java)를 커멘드 상 실행하기

```java
package com.ss.java_basic1

public class HelloWorld {
	public static void main(String[] args){
		System.out.println("Hello World");
	}
}
```

1. javac -d . [HelloWorld.java](http://helloworld.java) 로 class 생성
2. java 클래스 이름으로 실행


**Type**

- primitive : 정해진 크기의 memory size
- reference : 정해질 수 없음, 공간의 주소를 저장

<br>

**Type casting**

- 묵시적 : 큰 type → 작은 type
    - 정수형은 실수형으로 자동형변환
- 명시적 : 작은 type → 큰 type

`/** 코드 */`  : Javadoc 사용

**bit operator**

```java
public class Test {
	public static void main(String[] args) {
		
		// bit and
		System.out.println("3 & 4 = " + (3 & 4));
		//   0011
		// & 0100
		// ------
		//   0000
		
		// bit or
		System.out.println("3 | 4 = " + (3 | 4));
		//   0011
		// | 0100
		// ------
		//   0111
		
		// bit exclusive(xor)
		System.out.println("3 ^ 4 = " + (3 ^ 4));
		//   0011            
		// ^ 0100
		// ------
		//   0111
		
		// bit not
		System.out.println(" ~ 4 = " + (~ 4));
		//   0100
		// ~
		// ------
		//   1011
	}
}
```


## Bit 연산

- *,/ 연산자에 비해 처리 속도가 훨씬 빠름.

```java
public static void main(String[] args) {
		
		// <<
		System.out.println("1 << 2 = " + (1 << 2));
		//    .... 0000 0001
		// << .... 0000 0100         1 << 2 = 4
		
		System.out.println("-16 >> 2 = " + (-16 >> 2));
		//	  .... 1111 0000
		// >> .... 1111 1100           -16 >> 2 = -4
		
		// >>>
		System.out.println("7 >>> 2 = " + (7 >>> 2));
		//	   0000 .... 0000 0111
		// >>> 0000 .... 0000 0001       7 >>> 2 = 1
		
		System.out.println("-5 >>> 24 = " + (-5 >>> 24));
		//	   1111 1111 1111 1111 1111 1111 1111 1001
		// >>> 0000 0000 0000 0000 0000 0000 1111 1111

	  // -5 >>> 24 = -1
	}
```

**조건문**

- switch에서는 double 사용x ( String은 Java 버전 바뀜에 따라 됨)
- break 없으면 계속 내려가면서 검사
- 'A'와 65 둘 다 case에 있으면 겹치기 때문에 오류남.

**삼항 연산**

```java
public static void main(String[] args) {

		int N = 6;
		
		boolean isEven = ( N % 2 == 0 ) ? true : false;
		
		N = ( ! isEven ) ? 10 : 20;
		
		System.out.println(N); // 20
	}
```


## 데이터 타입 문제

**정수의 문제점 : 범위 문제(Overflow)**

```java
int i = Integer.MAX_VALUE; // 가장 큰 수 알아보기
int i2 = i + 1;
		
System.out.println(i); // 2147483647
System.out.println(i2);  // -2147483648 , Integer.MIN_VALUE 가 되버림.

long l1 = i+1;
long l2 = (long)(i+1); // 이미 깨진걸 사용해서 -2147483648가 나옴
		
long l3 = (long)i + 1; // 2147483648, 형 변환 후 연산
```

` 그렇지만 long도 무한대는 아니라 한계 존재...

    → 별도의 class(BigInteger 등)로 표현!


**실수의 문제점 : 부동소수점** 

- 컴퓨터는 정확한 실수를 표현하지 못함(근사값 사용)

    → 실수의 계산은 믿을만한 게 아님(오차 허용)

    → 정수로 올려서 계산하면 된다.

BigDecimal등을 이용할 수 있으나 무거움.

#### 값의 동등 비교

- 기본형 : ==
- 객체형 : equals method