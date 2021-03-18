---
title: "Graph 정리"
date: 2021-03-18T16:27:06+09:00
draft: false
categories : ["Algorithm", "Graph"]
---


# 그래프

`정점들의 집합과 이들을 연결하는 간선들의 집합

## 그래프 표현

**인접 행렬**

N*N의 행렬을 이용

- 인접된 부분에 연결되어 있으면 1을 주거나 가중치 부여
- 희소 행렬에서 비효율적

**인접 리스트**

정점에 대한 인접 정점들을 노드로 하는 연결 리스트로 저장

- 노드 추가, 삭제가 빈번한 경우 사용하기 좋음

**간선 리스트**

간선 자체를 객체로 표현하고 두 정점의 정보를 가짐

## 그래프 탐색

- BFS

    ```java
    void bfs(int start){
    	Queue<Integer> queue = new LinkedList<>();
      queue.add(start); // 생성, 삽입
    	visited[start] = true; //방문 표시

      int depth = 0;
      while (!queue.isEmpty()) {
          int size = queue.size();
          while (size-- > 0) {
              int head = queue.poll(); 
              sb.append(head).append(" "); // 결과 저장
              
    					// 연결된 모든 간선 탐색
              List<Integer> childs = graph2[head];
              for (int i = 0; i < childs.size(); i++) {
                  if (!visited[childs.get(i)]) { // 방문하지 않은 곳 탐색
                      visited[childs.get(i)] = true;
                      queue.add(childs.get(i)); // 방문 표시, 큐에 넣기
                  }
              }
          }
      }
    }
    ```

- DFS

    ```java
    void dfs(int node){
    	visited[node] = true; // 방문 표시
      // 사용
      sb.append(node).append(" "); // 결과 저장

      // 연결된 모든 간선 탐색
      List<Integer> nodes = graph2[node];
      for (Integer child : nodes) {
          if (!visited[child]) {
              dfs2(child); // 자식 재귀 탐색
          }
      }
    }
    ```

## 서로소 집합 (Disjoint-set)

- 중복 포함된 원소가 없는 집합들
- 집합에 속한 한 특정 멤버를 통해 집합을 구분 ⇒ 대표자
- 연결리스트나 트리, 배열로 disjoint-sets를 표현 가능
- 연산

    Make-Set(x) : 유일한 멤버 x를 포함하는 새로운 집합을 생성

    Find-Set(x) : x를 포함하는 집합을 찾음

    Union(x, y) : x와 y를 포함하는 두 집합을 통합 (x의 부모를 y의 부모로)

#### 문제점

depth가 너무 깊어지는 경우 → Rank, Path compression

Rank

- 각 노드가 자신을 루트로 하는 subtree의 높이를 Rank라는 이름으로 저장
- 두 집합을 합칠 때 rank가 낮은 집합을 랭크가 높은 집합에 붙이기
    - rank가 같을 경우, key 기준을 통해 판단. 정해진 집합은 rank 1 증가

Path compression

- Find-Set 과정에서 만나는 모든 노드들이 직접 루트를 가리키도록 포인터 바꾸기

```java
int findset(x){
	if(x==p[x]) return x;
	else return findset(p[x])
}

// path compression
int findset(x){
	if(x==p[x]) return x;
	else return p[x] = findset(p[x])
}
```

---


## 최소 신장 트리 (MST)

최소 비용 문제 : 두 정점 사이 최소 비용 경로, 간선들의 가중치의 합이 최소가 되는 트리 찾기

신장 트리 : n개 정점으로 이루어진 무향 그래프에서 n개의 정점과 n-1개 간선으로 이루어진 트리

**MST** : 무향 가중치 그래프에서 신장 트리를 구성하는 간선들의 가주치의 합이 최소인 신장 트리

- 사이클은 존재하면 안 됨

    ⇒ visit배열로 방문 여부를 기록

### Kruskal

1. 간선을 가중치에 따라 오름차순 정렬
2. 가중치가 가장 낮은 간선부터 선택하면서 트리 증가
    - 사이클이 존재하면 다음으로 가중치가 낮은 간선 선택
- Kruskal

    ```java
    void kruskal(g, w){
    	for v in g.v
    		make(v)

    		//g.e에 포함된 간선들을 가중치 w에 의해 정렬

    	for u,v in g.e
    		if findset(u) != findset(v)
    			union(u,v)
    }
    ```

### Prim

하나의 정점에서 연결된 간선 중 하나씩 선택

1. 임의 정점을 하나 선택
2. 인접 정점 중 최소 비용 간선의 정점 선택

- 트리 정점 : MST를 만들기 위해 선택된 정점

- 비트리 정점 : 선택X

위의 두 집합은 서로소 유지!

**사이클 여부 판단을 하지 않아도 됨** -> 애초에 사이클을 생성하지 않는다

- Prim

    ```java
    int prim(g, r){ // r은 시작 정점
    	for v in g.v
    		minedge[v] = inf
    	minedge[r]
    	while true
    		u = extract_min() // 방문하지 않은(mst 포함x) 최소 비용 인접 정점
    		visit[v] = true
    		result+=minEdge[v]
    		if(++cnt==N) break;	 // 모든 정점 연결
    		for u in g.e[v] // v 인접 정점들
    			if !visit[u] && w(v,u) < minedge[u] // v에서 u로의 최소비용 갱신
    				minedge[u] = w(v,u)
    	return result
    }
    ```
---
<br>

## 그래프 최단경로 알고리즘

bfs, dfs : 가중치가 없을 경우

#### 다익스트라
시작 정점이 주어지고 다른 모든 노드 사이의 최단거리 계산

1. 선택되지 않은 노드들 중 최단 거리가 가장 짧은 노드 선택
2. 선택된 노드와 연결된 다른 노드의 최단거리 갱신
3. 1번으로

가장 짧은 노드를 선택할 때 Priority queue 또는 heap을 사용하면 O(ElogE)

`dists[curr] = min(dists[curr], dists[next] + cost(curr,next)`

curr : 선택된 노드 , next : 갱신해야 하는 노드

```java
while (!pq.isempty()) {
	int curr = pq.top().second;
	pq.pop();
	for (node it : nodes[curr]) {
		int next = it.target_node;
		int cost = it.weight;
		if (dist[next] < 0 || dist[next] > dist[curr] + cost) {
			dist[next] = dist[curr] + cost;
			pq.push(pair<int, int>(dist[next], next));
		}
	}
}
```

#### 플로이드 와샬

2차원 배열, 복잡도 n^3

`dists[i][j] = min(dists[i][j], dists[i][k] + dists[k][j])`

- 노드 0부터 k를 경유할 수 있을 때 노드 i부터 j까지의 최단거리 정의
- 최단경로가 노드 k를 경유하거나 경유하지 않는 경우를 계산해 최소값이 최단경로

```java
for (int k = 0; k < nodes.size(); ++k) {
	for (int i = 0; i < nodes.size(); ++i) {
		for (int j = 0; j < nodes.size(); ++j) {
			dists[i][j] = min(dists[i][j], dists[i][k] + dists[k][j]);
		}
	}
}
```

#### 벨만 포드 
음수 간선이 있을 때(음의 사이클)

`dists[curr] = min(dists[curr], dists[next] + cost(curr,next)`

다익스트라와 동일한 식이나 모든 edge에 대해 적용

음의 사이클이 있을 수 있어 다익스트라 1번을 한번더 실행

```java
for (int i = 0; i < nodes.size() - 1; ++i) {
	for (int curr = 0; curr < nodes.size(); ++curr) {
		if (dist[curr] < inf) {
			for (auto it : nodes[curr]) {
				int next = it.target_node;
				int cost = it.weight;
				dist[next] = min(dist[next], dist[curr] + cost);
			}
		}
	}
}
```