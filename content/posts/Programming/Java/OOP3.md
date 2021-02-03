---
title: "OOP3"
date: 2021-01-31T18:12:07+09:00
draft: false
categories : ["Programming", "Java"]
---

` 상속과 다형성에 대해 알아보자

# Inheritance

B가 A의 member variables과 method를 그대로 받으면 상속받는다고 하고 부모-자식, 상위-하위 관계이다.

- 기존의 클래스에서 자산(변수,메서드)을 자식 클래스에서 재사용 → 코드의 절감
- 접근 제한자에 상관없이 상속되지만 자식에게 보이지 않을 뿐..
- 어떤 Class가 아무런 상속을 받지 않을 경우, 자동으로 `java.lang.Object.Class`가 Class의 부모가 된다.

```java
public class B extends A{
	...
}
```

자바는 다중 상속 불가능

**is a**

- extends 관계

**has a**

- 상속하지 못한다고 버릴 필요는 없고 멤버 변수로 가지고 있기

구분..? 

- is a 적합한지 보기 (ex superman is a person? superman is a super?)
- 나머지는 has a


**super** : 조상 class의 생성자 호출

- super(name) 처럼 생성자로 넘기는 식으로 활용 가능
- this를 통해서 가지고 있는 것과 상속받은 것을 확인할 수 있다.

**this는 나의 다른 생성자를 부를 때도 사용**

**주의사항**

- 둘 다 첫 줄에 사용해야 함

    → this와 super를 같이 쓸 순 없음

다음은 가능

```java
public Corona(String name, int level, int spreadSpeed){
	super(name,level);
	this.spreadSpeed = spreadSpeed;
}
```

- 명시적으로 this 또는 super를 호출하지 않으면 생성자의 첫 줄에는 super() 가 생략되어 있음.

- 하위 class에서 private, default로 접근할 수 없기 때문에 접근해야 할 것은 protected사용하거나 public method활용

---


|상속||
|:-:|:-:|
|Object|toString|
|Virus|-|
|Corona|toString|
|ChildCorona|-|

