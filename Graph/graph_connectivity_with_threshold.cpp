1627. Graph Connectivity With Threshold
========================================
// We have n cities labeled from 1 to n. Two different cities with labels x and y 
// are directly connected by a bidirectional road if and only if x and y share 
// a common divisor strictly greater than some threshold. More formally, cities 
// with labels x and y have a road between them if there exists an integer z 
// such that all of the following are true:

//     x % z == 0,
//     y % z == 0, and
//     z > threshold.

// Given the two integers, n and threshold, and an array of queries, you must 
// determine for each queries[i] = [ai, bi] if cities ai and bi are connected 
// directly or indirectly. (i.e. there is some path between them).

// Return an array answer, where answer.length == queries.length and answer[i] 
// is true if for the ith query, there is a path between ai and bi, or answer[i] 
// is false if there is no path.

 

// Example 1:

// Input: n = 6, threshold = 2, queries = [[1,4],[2,5],[3,6]]
// Output: [false,false,true]
// Explanation: The divisors for each number:
// 1:   1
// 2:   1, 2
// 3:   1, 3
// 4:   1, 2, 4
// 5:   1, 5
// 6:   1, 2, 3, 6
// Using the underlined divisors above the threshold, only cities 3 and 6 share a 
// common divisor, so they are the
// only ones directly connected. The result of each query:
// [1,4]   1 is not connected to 4
// [2,5]   2 is not connected to 5
// [3,6]   3 is connected to 6 through path 3--6

// Example 2:

// Input: n = 6, threshold = 0, queries = [[4,5],[3,4],[3,2],[2,6],[1,3]]
// Output: [true,true,true,true,true]
// Explanation: The divisors for each number are the same as the previous example. 
// However, since the threshold is 0,
// all divisors can be used. Since all numbers share 1 as a divisor, all cities 
// are connected.

// Example 3:

// Input: n = 5, threshold = 1, queries = [[4,5],[4,5],[3,2],[2,3],[3,4]]
// Output: [false,false,false,false,false]
// Explanation: Only cities 2 and 4 share a common divisor 2 which is strictly 
// greater than the threshold 1, so they are the only ones directly connected.
// Please notice that there can be multiple queries for the same pair of nodes 
// [x, y], and that the query [x, y] is equivalent to the query [y, x].



// This problem is about connect pair of cities then check if there is a connection 
// between any 2 cities. This is clearly Union-Find problem. 
// The difficult things is that n <= 10^4, we can''t check all combination pairs 
// of cities which is O(n^2) which will cause TLE. 

// using Sieve of Eratosthenes time complexity  ~ O(NlogN)

// Input: n = 6, all pair of cities:

// Let z is the common divisor strictly greater than some threshold
// z = 1, x = [2, 3, 4, 5, 6]
// z = 2, x = [4, 6]
// z = 3, x = [6]
// z = 4: x = []
// z = 5, x = []
// z = 6, x = []


// Time: ((m + n)logn)

// Union (i, j), wherex = kz, where 2 ≤ k ≤ n / 2: O(nlogn) = n + n/2 + n/3 + ... + 1
// Queries: O(mlogn), m is number of queries, uf.find(x) takes time of logn

// Space: O(m + n)

//     Answer list: O(m)
//     Union find parent array: O(n)


class UnionFind{
  vector<int> parent,rank;
  public:
    UnionFind(int n){
        parent.resize(n);
        rank.resize(n);
        for(int i=0;i<n;++i){
            parent[i]=i;
            rank[i]=1;
        }
    }
    int find(int x){
        if(parent[x]==x) return x;
        return parent[x]=find(parent[x]);
    }
    bool unite(int x,int y){
        int px=find(x),py=find(y);
        if(px==py) return false;
        if(rank[px]>rank[py]){
            rank[px]+=rank[py];
            parent[py]=parent[px];
        }
        else{
            rank[py]+=rank[px];
            parent[px]=parent[py];
        }
        return true;
    }  
};
class Solution {
public:
    vector<bool> areConnected(int n, int threshold, vector<vector<int>>& queries) {
        
        UnionFind uf(n+1);
        for(int i=threshold+1;i<=n;++i)
            for(int j=2*i;j<=n;j+=i)
                uf.unite(i,j);
        vector<bool> res;
        for(auto &q:queries){
            res.push_back(uf.find(q[0])==uf.find(q[1]));
        }
        return res;
    }
};


class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1] * n

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, u, v):
        pu = self.find(u)
        pv = self.find(v)
        if pu == pv: return False
        if self.size[pu] > self.size[pv]:  # Union by larger size
            self.size[pu] += self.size[pv]
            self.parent[pv] = pu
        else:
            self.size[pv] += self.size[pu]
            self.parent[pu] = pv
        return True

class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        uf = UnionFind(n + 1)
        for z in range(threshold + 1, n + 1):
            for x in range(z + z, n + 1, z):  # step by z
                uf.union(z, x)

        ans = [False] * len(queries)
        for i, (u, v) in enumerate(queries):
            ans[i] = uf.find(u) == uf.find(v)
        return ans


//compact

int find(vector<int> &ds, int i) {
    return ds[i] < 0 ? i : ds[i] = find(ds, ds[i]);
}
vector<bool> areConnected(int n, int threshold, vector<vector<int>>& queries) {
    vector<int> ds(n + 1, -1);
    for (int i = threshold + 1; 2 * i <= n; ++i) {
        if (ds[i] != -1)
            continue;
        int p1 = i;
        for (int j = 2 * i; j <= n; j += i) {
            int p2 = find(ds, j);
            if (p1 != p2) {
                if (ds[p1] > ds[p2])
                    swap(p1, p2);
                ds[p1] += ds[p2];
                ds[p2] = p1;
            }
        }
    }
    vector<bool> res;
    for (auto &q : queries)
        res.push_back(find(ds, q[0]) == find(ds, q[1]));
    return res;
}