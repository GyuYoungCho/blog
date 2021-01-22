---
title: "OOP1"
date: 2021-01-22T23:12:07+09:00
draft: false
categories : ["Algorithm", "Java"]
---

# OOP

` 우리는 객체 지향적 삶을 살고 있고 그러한 현실을 프로그래밍에 반영하려고 함.

**장점**

- 객체 교체(유지 보수)에 좋음
- 재사용성

## OOP의 특징

**OOP is A P.I.E**

- Abstraction : 특징 추출

    → 현실의 객체(프로그램의 대상으로 삼는 것)를 추상화해서 class를 만들고 이를 구체화해서 object를 만든다.

- Encpsulation : 필요한 기능 공개

    → 데이터의 은닉과 보호

- inheritance : 상속

    → member variables, method를 자식이 물려받음 → 재사용

- Polymorphism : 다형성

    → 하나의 객체를 여러 타입으로 참조하는 것

## Class

**class :** 현실 세계를 추상화 해놓은 것  ex) 설계도, 청사진 등

→ object를 만들기 위해 필요하지만 직접 class를 사용하진 않음.

**객체(object,instance)** : 클래스를 구체화해서 메모리에 생성된 것

**class = type**

**factor**

- attribute : member variables → 각각 객체마다 다를 수 있음
- behavior : methods → 동작, 객체들마다 같음
- Constructor(생성자) → member variable 초기화

**new**

Constructor를 보고 memory allocation 수행

- 생성자는 기본부터 여러 parameter를 가지는 등 다양하게 생성 가능

**member variables**

- 다양한 상태 표현
- 설정하지 않으면 default value 부여
- OOP적 관점에서 외부에서 값을 바꾸는 것은 좋지 않음

---

## Encapsulation

- member variables과 method를 필요한 경우를 제외하고 노출하지 않도록 가정
- 노출할 경우 set&get 으로 소통
- 여전히 외부에서 접근가능하므로 private 설정


- new는 heap 영역에 객체를 생성한다는 의미!

```java
int i1 = 10;
int i2 = 10;
		
String s1 = "Hello";
String s2 = "Hello";
// s1과 s2는 같은 곳을 가리킴
String s3 = new String("Hello");
String s4 = new String("Hello"); //new를 통해 생성하면 따로 생성됨

// == 는 메모리값 비교함
if( i1 == i2 ) { System.out.println("i1 i2 Same"); }
if( s1 == s2 ) { System.out.println("s1 s2 Same"); }
if( s3 == s4 ) { System.out.println("s3 s4 Same"); } //equals 쓰면 나옴
```

--- 


## String class

**StringBuilder**

+를 사용하면 불필요한 객체가 많아져 사용

loop 등에서는 stringBuilder가 효과적

append를 이용

```java
StringBuilder sb = new StringBuilder("");
sb.append(s1).append(", ").append(s2);
```

**toString 사용**

- toString이 자동으로 생성되고 재정의해서 사용

```java
public String toString() {
		return this.name + " " + this.color + " " + this.price;
	}
```

```java
// main
System.out.println(phone); // Galaxy Note B 10000
```

**값 전달**

```java
public class PassByValueTest {
	public static void main(String[] args) {
		int i = 10;
		setVal(i); // 값 전달,최종적으로 안 바뀜.
		System.out.println(i);
		
		Pass p = new Pass();
		p.val = 10;
		setVal(p); // reference 전달, 주소값을 찾아 값이 5로 바뀜.
		System.out.println(p.val);
	}
	
	public static void setVal(int x) { x = 5; }
	
	public static void setVal(Pass p) { p.val = 5; }
}

class Pass{
	public int val = 3;
}
```

## Package

- 같은 패키지일 경우 package 선언하면 사용가능
- 다른 패키지일 경우 import로 불러와야 함. (com.web.*)

## Access Modifier

|구분|Same Class|Same Package|Sub Class|Universe|
|-:|:-:|:-:|:-:|:-:|
|private|O|X|X|X|
|default|O|O|X|X|
|protected|O|O|O|X|
|public|O|O|O|O|
