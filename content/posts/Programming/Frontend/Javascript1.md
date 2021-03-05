---
title: "JavaScript1"
date: 2021-03-05T19:12:07+09:00
draft: false
categories : ["Programming", "Frontend"]
---

# JavaScript

### 선언

`<body>` 안에 위치하면 브라우저가 html부터 해석하여 화면에 그리기 때문에 빠르다고 느낄 수 있어서 보통 body안 맨 밑에 삽입하는 경향임. 

`html과 연결하기

```html
외부
<script language="text/javascript" src="외부 스크립트 파일의 URL"></script>
내부
<script type="text/javascript"></script>
```

#### 데이터 타입

primitive

숫자, 문자열, boolean, undefined(초기화되지 않음), null(없거나 빔)

reference

window(브라우저 및 창 프레임 표시), document(html 파일 기술된 문서 제공 및 작업 수행), Date, Array, RegExp(정규 표현식)

<br>

#### 변수

var로 선언, 값 대입 시점에 타입 자동 설정, 중복 선언 가능

- 전역 변수 : 외부에서 선언 후 함수 내에서 var 없이 사용
- 지역 변수 : 내부에서만 사용

let : 중복을 허용하지 않음

프로그래밍의 안전성을 위해 let을 쓰는 게 더 나음.

const : 상수, 바꿀 수 없음.

앞에 선언 없이 변수 선언 : 전역 레벨 변수 선언 → 절대 하지 마요...

→ 전역 객체인 window의 속성으로 등록해버림.

<br>

**Primitive Type**

1. 숫자 자료형 : 모든 숫자를 실수로 처리

	- js에서는 연산에서 예외를 발생시키지 않음
	- Infinity : 무한대
	- NaN : 결과가 숫자가 아님

2. string : 따옴표 상관x, 16비트
3. boolean, null. undefined 

- 빈 문자열,  null, undefined, 숫자 0은 false로 간주됨.

js에서는 자료형에 대해 느슨한 구조 → 어떤 자료형이든 전달하고 변환 가능

<br>

**사용자 정의 함수**

```jsx
function 함수명([arg1, arg2, ...]) {
	statement
	[return expression;]
}
```

#### Hoisting

js에서 일어나는 현상, js는 parsing과 실행 2단계로 처리됨

- parsing : 전역 레벨의 var 변수 인지 및 undefined 초기화, 함수에 대해 함수명과 동일한 변수 생성 및 인지
- 실행 : 할당 등 실행문 실행

선언과 초기화를 먼저 하기 때문에 선언되기 전에 참조 가능

예시

```jsx
console.log("1", num1, square, square2);
var num1 = 10;
console.log("2", num1, square, square2);
// 전역 레벨의 변수 선언 - 절대 하지 않기..
// num2 = 20;

var square = 0;
console.log("3", num1, square, square2);
function square(x){
	return x*x
}
// square(num1); //square가 함수가 아니라 안됨.
var square2 = function(){
	return x*x
}
console.log("5",num1, square, square2);

결과

1 undefined ƒ square(x){
	return x*x
} undefined
2 10 ƒ square(x){
	return x*x
} undefined
3 10 0 undefined
5 10 0 ƒ (){
	return x*x
}
```

let으로 선언할 경우 1,2,3처럼 할당되지 않은 것이 실행되버리는 결과를 막을 수 있음.

<br>

#### 연산자

===, !== : 일반적인 ==과 !=가 "1" 과 1도 같다고 판별하기 때문에 타입까지 비교하기 위함.

- \+ : 문자열 결합연산, 0(false)/1(true) , 참조형일 경우 toString()결과와 결합
- -, *, /,% : 문자열을 최대한 숫자로 바꾸고 못하면 NaN, 참조형일 경우 valueOf()의 결과와 연산

논리 연산 : 마지막으로 평가한 값을 리턴

```jsx
console.log(1 + "123")
console.log(true + true + false)
console.log(new Date() + 123)
//1123
//2
// 현재 시간123

console.log(1 - "123")
console.log(0 - "A")
console.log(new Date().valueOf())
console.log(new Date()-60*60*24*1000);
//-122
//NaN
//1614832363259
//1614745963259

console.log(true==1,true===1, null==undefined, null===undefined)
//true false true false

let result1 = 1 || undefined; // true || false -> true
console.log(result1);

let result2 = 0 || undefined; // false || false -> false
console.log(result2);

let result3 = null || "False"; // false || "False" --> "False"
console.log(result3);
if(result3){ // true임
	console.log("Hello js")
}
// 1
// undefined
// False
// Hello js

```

#### 반복문

of : 값이 바로 나오게

in : 배열 순회 보다는 객체의 속성 탐색

```jsx
let arr = [1,2,3,4,5];
for (var i = 0; i < arr.length; i++) {
	console.log(arr[i])	
} //12345

for(let i in arr){ // i는 index라 할 수 있음.
	console.log(i)
} //01234

for(let i of arr){ // data 자체가 나옴
	console.log(i) 
} //12345

let person = {
		name : "홍길동",
		age : 30 
}
for(let key in person){
	console.log(key,person[key]);
}
```

## 객체 생성

**객체 리터럴**

```jsx
var a = { width: 20, height: 30, position: {x:200, y:400} };
```

**Object 생성자 함수**

```jsx
var a = new Object();
a.name = "가나다"
a.age = 30;
a.info = function(){
	console.log("hi")
};
```

**생성자 함수**

```jsx
function a(name,age){
	this.name = name;
	this.age = age;
	this.info = function(){
		console.log("hi")
	};
}
```

---

## Window 객체

웹 브라우저에서 작동하는 js 최상위 객체(BOM)

함수를 호출하면 브라우저에서 제공하는 창 open

- alert : 알림창
- confirm : 확인/취소 선택창
- prompt : 입력창 + 확인/취소

**navigator**

브라우저 정보가 내장된 객체, 서로 다른 브라우저 구분, 다르게 처리 가능

user-agent로 mobile환경, chrome 등 정보 확인 가능

**location**

url정보 관련

- location.href : 현재 url 조회. 값을 할당하면 해당 url로

- reload : 새로고침

**history**

페이지 이력을 담는 객체

- back, forward : 뒤로 가기, 앞으로 가기

- window.open(url,창이름,특성,히스토리 대체여부)

	- 특성 : 창의 너비 높이, 위치, 크기 조정, menubar 등.

	- 대체여부 : 현재 페이지 히스토리에 덮어쓸지 여부

```jsx
function windowOpen() {
	window.open('./index.html', 'abc', 'width=300, height=200');
}
```

---

# DOM

문서의 구조를 정의하는 API제공

### 문서 조작

- createElement(name) : 엘리먼트생성

    ```jsx
    var ele = document.createElement("img"); // 메모리에 생성
    ```

- append(string | node) : 엘리먼트 추가

    ```jsx
    parent.append(element);
    ```

- setAttribute(name, value) : 속성 변경. ,getAttribute(name) : 값 가져옴

    ```jsx
    ele.setAttribute("width", 200);
    ele.width = 200;
    ```

    사용자 속성 변경시에는 setAttribute를 써야 함

    ```jsx
    ele.setAttribute("name", "차"); => 성공

    ele.name = "차"; => 실패
    ```

- innerHTML : 요소 내용 변경

    ```jsx
    list.innerHTML = "<img src='./images/cake.jpg' width='200'/>"
    ```

### 문서 접근

- getElementById(String)

    ```jsx
    var ele = document.getElementById("a"); => <div id="a">지역</div>
    ```

- querySelector(css selector)

    ```jsx
    var ele = document.querySelector("#a") => <div id="a">지역</div>
    var ele = document.querySelector("div") => <div id="a">지역</div>
    var ele = document.querySelector(".b") => <div class="b">지역</div>
    var ele = document.querySelector("[name='c']") => <div name="c">구미</div>
    ```

- querySelectorAll(css selector) : 결과를 배열처럼 사용

    ```jsx
    var list = document.querySelectorAll("div");
    for (var i=0; i<list.length; i++) {
    	console.log(list[i])
    }
    ```

    ```jsx
    var ele = document.querySelectorAll("div"); 
    => <div id="a">지역</div>
    	 <div id="b">광주</div>
    	 <div id="c">구미</div>

    var ele = document.querySelectorAll(".b");
    => <div id="a" class="b">지역</div>
    	 <div class="b">광주</div>

    ```

    ```jsx
    ele.width = 200; // 사용자 속성에는 접근 불가
    ele["width"] = 200; // 사용자 속성에 접근 가능!!!!
    ```

---

## 이벤트 처리

요소.addEventLister( 이벤트 타입, 이벤트리스너(함수명), 이벤트전파방식);

```jsx
tn.addEventListener("click", doAction, true);
```

localstage에는 문자열 밖에 저장되지 않음.

---

### 기타

#### 문자열 표현

```jsx
console.log("이름은"+person.name+"나이는" + person.age);
console.log(`이름은 ${person.name}, 나이는 ${person.age}`);
```

#### JSON

client ←→ server 간 데이터 전달 -> JSON 활용

JSON(JavaScript Object Notation) : js의 객체 표현법

JSON.stringify(object) : json → string으로 

JSON.parse(str) : string → json

<BR>

**function의 다양한 용도**

1. 일반적인 method 역할
```jsx
function sayHi(){
	console.log("Hi");
}
sayHi();
```
2. 객체로써의 역할
```jsx
let sayHello = function(){
	console.log("Hello");
}
sayHello();
```
3. 생성자로써의 역할
```jsx
function Student(name, age){
	this.name = name;
	this.age = age;
}
let student = new Student("홍길동",30);
```


<br> 

**callback**

```jsx
// 기준을 가지고 정렬해보기
strs.sort(function(first, second){
	return first.length - second.length;
})

// 3초에 한 번씩 함수 실행하게 하기
window.setInterval(function() {
	console.log(new Date());
}, 3000)

// 10초 후
window.setTimeout(function() {
	location.href = "https://www.google.com";
}, 10*1000)
```

일정 간격으로 함수를 일정 조건만큼 실행하고 싶은 경우, 재귀 활용

for문 안에 그냥 쓸 경우 setTime으로 딜레이 되어도 for문은 계속 돌아가서 결국 동시에 실행되는 것처럼 보임 

→ setTimeout안에서 계속 실행되어 일정 시간 후 함수가 실행된 후 그 안의 setTimeout이 실행된다고 할 수 있음!

```jsx
function looper(i) {
	if (i < 6) {
		setTimeout(function () {
			one_choice(lotto[i])
			i++;
			looper(i);
		}, 2000)
	}
}
looper(0);
```