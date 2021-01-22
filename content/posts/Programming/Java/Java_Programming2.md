---
title: "Java_Programming2"
date: 2021-01-21T19:12:07+09:00
draft: false
categories : ["Algorithm", "Java"]
---

# 배열

#### for each Array

```java
int arr [] = {1,2,3,4,5};

for(int x : arr){
    System.out.println(x);
}
```

#### Array is immutable

- 크기 변경 불가
- 변경이 필요할 경우 새로 작성

**arraycopy**

```java
String [] students = { "홍길동", "박사", "윤식당", "나오기" };
String [] students3 = new String[5];
System.arraycopy(students, 0, students3, 0, 4); //[홍길동, 박사, 윤식당, 나오기, null]
// index와 length를 정해 copy
```

```java
int[] srcArray = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int[] tgtArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		
System.arraycopy(srcArray, 2, tgtArray, 1, 3); // [0, 2, 3, 4, 0, 0, 0, 0, 0, 0]
```


```java
int[] intArray = { 3, 27, 13, 8, 235, 7, 22, 9, 435, 31, 54 };
			
int min = Integer.MAX_VALUE;
int max = Integer.MIN_VALUE;
			
for (int i = 0; i < intArray.length; i++) {
	min = Math.min(min, intArray[i]);
	max = Math.max(max, intArray[i]);
}
```

**valuecount**

```java

int[] intArray = { 3, 7, 2, 5, 7, 7, 9, 2, 8, 1, 1, 5, 3 };
int[] used = new int[10];

for (int i = 0; i < intArray.length; i++) {
	used[intArray[i]]++;
}

```


## 2차원 배열

**array 만들기**

- `int [][]Array = new int[4][3];`
- `int [][] intArray5 = new int[][] {{1,2,3},{1,2,3},{1,2,3},{1,2,3}};`

```java
int  intArray[][] = new int [4][];
intArray[0] = new int[2];
intArray[2] = {1,2,3}; // 안됨
```

- InputStreamReader와 BufferedReader를 이용해 입력처리를 빠르게 할 수 있음.


---

## Array memory

```java
static void makeAndPrint() {
		// 로컬 영역
		int [] arr1 = new int[3];
		int [] arr2 = {1,2,3} // array constant

// arr2 = {4,5,6} 불가능
		arr2 = new int[] {4,5,6};
// 기존의 공간이 사라지고 새로운 공간이 할당됨.
// 기존의 공간은 누구도 참조하지 않고 GC가 자동으로 회수
}
```

` stack → local　　　　　 heap → 객체

- `int [] arr1` : arr1 (참조형) 생성
- `new int[3]` : 3개의 int를 저장할 공간 생성(heap에 만들기!)
    - new로 생성할 경우 얼마나 공간을 차지할지 알려줘야 함.
- 32bit*3의 공간의 주소가 stack에 저장된다.

자바는 GC가 자동으로 사용하지 않는 메모리를 회수함


## n차원 배열

**2차원** 

: 1차원 배열을 관리하는 1차원 배열이라 할 수 있음

`int [][] arr3 = new int[3][]`

- arr3의 3개의 공간에 각각 `int[]`가 들어가야 함.(기본값 null)

```java
arr3[0] = new int[]{1,2,3,4,5};
arr3[1] = new int[]{6,7};
arr3[2] = new int[]{8,9};
```

## Reference

```java
static void reference() {
		char [] chars = "Hello".toCharArray();
		change1(chars[0]); // 값이 복사되어 change로 넘어감.
		System.out.println(Arrays.toString(chars)); 
// 실제 H는 바뀌지 않았음
		change2(chars);
// chars의 주소값을 넘겼기 때문에 바뀌게 된다.
}

static void change1(char data) {
	System.out.println(data+32); // 104
	data+=32; // 단항 연산자에서는 형변환이 일어나지 않음.
	System.out.println(data); // h
}

static void change2(char[] c2) {
		c2[0]+=32;
}
```

**엄밀히 말하면 위에서 chars는 배열이기 보다는 배열을 가리키는 주소값이라 할 수 있음**

**swap**

```java
static void swapTest() {
		int[] arr = {1,2,3};
		swap(arr,1,2); // arr의 주소가 넘겨짐
}
	
static void swap(int[] arr ,int a, int b) {
		int t = arr[a];
		arr[a] = arr[b];
		arr[b] = t; // 실제로 값이 바뀜.
}
```



## Arrays class

- 배열을 사용할 때 유용한 기능 제공

**배열의 제약사항**

- 타입, 크기

```java
char [] chars = "Hello Ssafy 5th class 12!!".toCharArray();
char [] largeOne = Arrays.copyOf(chars, 40);
// 한꺼번에 복사

char [] copy2 = Arrays.copyOfRange(chars, 0, 5);
// 일부분 복사

Arrays.fill(largeOne, '#');
// 특정 값으로 채우기, 초기화에 유용
		
Arrays.sort(chars);
// 정렬
```

---

# Array Delta Traversal

여태까지는 indexing을 이용한 방법

방향을 나타내는 delta 행렬 선언

```java
static int[][] deltas = { { -1, 0 }, { 0, 1 }, { 1, 0 }, { 0, -1 } };
// 상하좌우를 나타내는 Delta 배열
static int[][] deltaPlus = { { -1, -1 }, { -1, 1 }, { 1, 1 }, { 1, -1 } };
// 대각선
```


```java
for(int r=0;r<3;r++) {
	for(int c=0;c<3;c++) {
		int sum = 0;
		for(int d=0;d<4;d++) {
			int nr = r + deltaPlus[d][0];
			int nc = c + deltaPlus[d][1];
			if(isIn(nr,nc))
				sum +=map[nr][nc];
		}
		result[r][c] = sum;
	}
}
```