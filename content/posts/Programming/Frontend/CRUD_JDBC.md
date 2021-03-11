---
title: "CRUD and JDBC"
date: 2021-03-11T23:12:07+09:00
draft: false
categories : ["Programming", "Frontend"]
---

# DML

## Insert

```sql
insert into member (name1, name2, ....)
values (val1, val2...);
```

` Auto increment : primary key에서 자동으로 데이터 개수를 알 수 있도록

## Update

```sql
update table set name1 = val1, [name2 = val2...namen = valn]
where name = 'a'
```

- Where 생략하면 모든 데이터가 바뀜

## Delete

```sql
delete from table
where name = 'a'
```

## Select

```sql
select a from table where condition
```

 * : 모든 열

distinct : 중복 행 제거

expression, alias 등

ifnull(expr1, expr2) : expr1이 null이면 expr2 출력

case..then : 조건에 대한 값을 바꿔서 보여줘야 할 때

```sql
select salary
	case when salary > 15000 then '고액'
			 when salary > 8000 then '평균'
			 else '저액'
```

`where id in ('a','b','c')` : 안에 포함되는지 여부

`where salary between 60 and 100` : 사이값 

`where salary is not null` : null 아닌거 나오게 

Like

```sql
--A로 시작하는 문자를 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE 'A%'

--A로 끝나는 문자 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE '%A'

--끝에서 3번째 자리가 A인 문자 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE '%A__'

--A를 포함하는 문자 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE '%A%'

--A로 시작하는 두글자 문자 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE 'A_'

--첫번째 문자가 'A''가 아닌 모든 문자열 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE'[^A]'

--첫번째 문자가 'A'또는'B'또는'C'인 문자열 찾기--
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE '[ABC]'
SELECT 컬럼명 FROM 테이블 WHERE 컬럼명 LIKE '[A-C]'
```

논리 연산시 주의

- NOT NULL → NULL
- NULL AND TRUE → **NULL**,  NULL AND FALSE → FALSE
- NULL OR FALSE → **NULL**, NULL OR TRUE → TRUE

order by : 정렬(default : ASC), DESC : 내림차순

---

# JDBC API

### JDBC 연결 순서

1. Connection  생성 : com.mysql.cj.jdbc.Driver 로딩
    - Connection은 인터페이스임

    ```java
    Connection con = null;
    ```

2. 연결

    ```java

    con = DriverManager.getConnection(
    				"jdbc:mysql://127.0.0.1:포트번호/db이름?serverTimezone=UTC&useUniCode=yes&characterEncoding=UTF-8", 
    				id, pwd
    			);
    ```

3. statement 생성

    ```java
    Statement st = con.createStatement();
    ```

4. execute() or exectueQuery() 실행

    ```java
    rs = st.executeQuery("select * from emp");
    while(rs.next()) {
    		System.out.println(rs.getString("ename")+" , "+rs.getInt("sal"));
    }
     
    ```

5. preparedStatement
6. resultSet
- 담아서 커서를 통해 접근

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

// 기능만 제공한다 ->  멤버 변수, 즉 state가 없다  -> stateless
public class DBUtil {
	private static DBUtil util = new DBUtil();

	public static DBUtil getUtil() {
		return util;
	}

	private DBUtil() {
		try {
			// 1. Driver Loading
			Class.forName("com.mysql.cj.jdbc.Driver");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	// 2. Connection
	public Connection getConnection() {
		String url = "jdbc:mysql://127.0.0.1:포트번호/db이름?erverTimezone=UTC&useUniCode=yes&characterEncoding=UTF-8";
		Connection con = null;
		try {
			con = DriverManager.getConnection(url, id, pwd);
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return con;
	}

	public void close(ResultSet rset, Statement pstmt, Connection con) {
		try {
			if(rset!=null) {
				rset.close();
			}
			if(pstmt!=null) {
				pstmt.close();
			}
			if(con!=null) {
				con.close();
			}
		}catch(SQLException e) {
			e.printStackTrace();
		}
	}
}

private void selectTest(int idx) {
	Connection con = util.getConnection();
	// 3. Statement Create
	PreparedStatement pstmt = null;
	ResultSet rset = null;
	try {
		String sql = "select * from member where idx=?";
			
		// idx는 pk이므로 반복문 쓸 필요x..
		// 4. SQL Prapare and Execute 
		pstmt = con.prepareStatement(sql);
		pstmt.setInt(1, idx);
		rset = pstmt.executeQuery();
		if(rset.next()) {
			String userid = rset.getString("userid");
			String username = rset.getString("username");
			String userpwd = rset.getString("userpwd");
			String emailid = rset.getString("emailid");
			System.out.println(userid +" : " + username +" : " + userpwd+" : " + emailid);
		}
	}catch(SQLException e) {
		e.printStackTrace();
	}finally {
		//5. close
		util.close(rset, pstmt, con);
	}
}

```

Statement는 사용하지 않음 → sql injection 삽입공격 때문

- statement 는 sql을 그대로 전달
- preparedstatement는 값들을 인자처럼 처리