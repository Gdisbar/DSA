Building Bridges
====================
// Your are given two array north-bank a(i) & south-bank b(i)
// You want to connect as many north-south pairs of cities as possible with bridges 
// such that no two bridges cross. When connecting cities, you can only connect 
// city a(i) on the northern bank to city b(i) on the southern bank.

// Input : 6 4 2 1
//         2 3 6 5
// Output : Maximum number of bridges = 2
// Explanation: Let the north-south x-coordinates
// be written in increasing order.

// 1  2  3  4  5  6 //north
//   \  \
//    \  \        For the north-south pairs
//     \  \       (2, 6) and (1, 5)
//      \  \      the bridges can be built.
//       \  \     We can consider other pairs also,
//        \  \    but then only one bridge can be built 
//         \  \   because more than one bridge built will
//          \  \  then cross each other.
//           \  \
// 1  2  3  4  5  6 //south

// Input : 8 1 4 3 5 2 6 7 
//         1 2 3 4 5 6 7 8
// Output : Maximum number of bridges = 5

// north-south coordinates
// of each City Pair
struct CityPairs
{
    int north, south;
};
 
// sort in increasing order 1st south then north
bool compare(struct CityPairs a, struct CityPairs b)
{
    if (a.south == b.south)
        return a.north < b.north;
    return a.south < b.south;
}
 
// function to find the maximum number of bridges that can be built
int maxBridges(struct CityPairs values[], int n)
{
    int lis[n];
    for (int i=0; i<n; i++)
        lis[i] = 1;
         
    sort(values, values+n, compare);
     
    // longest increasing subsequence applied on the northern coordinates
    for (int i=1; i<n; i++)
        for (int j=0; j<i; j++)
            if (values[i].north >= values[j].north && lis[i] < 1 + lis[j])
                lis[i] = 1 + lis[j];
         
         
    int max = lis[0];
    for (int i=1; i<n; i++)
        if (max < lis[i])
            max = lis[i];
     
    // required number of bridges
    // that can be built       
    return max;       
}

struct CityPairs values[] = {{6, 2}, {4, 3}, {2, 6}, {1, 5}};


934. Shortest Bridge
========================
// You are given an n x n binary matrix grid where 1 represents land and 0 
// represents water.

// An island is a 4-directionally connected group of 1's not connected to any 
// other 1's. There are exactly two islands in grid.

// You may change 0's to 1's to connect the two islands to form one island.

// Return the smallest number of 0's you must flip to connect the two islands.

 

// Example 1:

// Input: grid = [[0,1],[1,0]]
// Output: 1

// Example 2:

// Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
// Output: 2

// Example 3:

// Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
// Output: 1


class Solution {
private:
    int row,col;
    queue<pair<int,int>> q; //<row,col>
    vector<vector<bool>> vis;
    vector<vector<int>> dir={{0,1},{0,-1},{1,0},{-1,0}};
    bool inValid(int x,int y){
        return (x<0||y<0||x==row||y==col);
    }
    
    void dfs(int x,int y,vector<vector<int>>& grid) {
        if (inValid(x,y)||grid[x][y]==0||vis[x][y]) return;
        q.push({x, y});
        vis[x][y]=true;
        dfs(x+1,y,grid);
        dfs(x-1,y,grid);
        dfs(x,y+1,grid);
        dfs(x,y-1,grid);
    }
    int bfs(vector<vector<int>>& grid){
        int res=0;
        while(!q.empty()){
            int sz=q.size();
            for(int i = 0;i<sz;++i){
                auto[x,y]=q.front();
                q.pop();
                for(int j=0;j<4;++j){
                    x=x+dir[j][0];
                    y=y+dir[j][1];
                    if(inValid(x,y)||vis[x][y]) continue;
                    if(grid[x][y]) return res; //reached Island
                    q.push({x,y}); //water,not our res
                    vis[x][y]=true;
                }
            }
            ++res; // end of each layer
        }
        return res;
    }
public:
    int shortestBridge(vector<vector<int>>& grid) {
        row = grid.size(),col=grid[0].size();
        vis.resize(row,vector<bool>(col,false));
        
        for(int i = 0;i<row;++i){
            for(int j = 0;j<col;++j){
                if(grid[i][j])
                    dfs(i,j,grid);
            }
        }
        return bfs(grid);
    }
};

//working
//queue<pair<int, int>> q
// if A[x,y]=0 assign them to A[x,y]=-1
// A[x,y] = -1 , skip changing A[x,y]

size : 1 i : 0 j : 1
**************************
x : 0 y : 2 A[x,y] : -1
x : 1 y : 1 A[x,y] : -1
x : 0 y : 0 A[x,y] : -1
===============================
size : 3 i : 0 j : 2
**************************
x : 1 y : 2 A[x,y] : -1
size : 2 i : 1 j : 1 
**************************
x : 2 y : 1 A[x,y] : -1
x : 1 y : 0 A[x,y] : -1
size : 1 i : 0 j : 0
**************************
===============================
size : 3 i : 1 j : 2
**************************







class Solution {
 public:
  int shortestBridge(vector<vector<int>>& A) {
    const int n = A.size();
    const vector<int> dirs{0, 1, 0, -1, 0};
    //vector<vector<int>> dirs={{0,1},{0,-1},{1,0},{-1,0}};
    int ans = 0;
    queue<pair<int, int>> q;
    // similar to union
    // mark one group to 2s by DFS and push them to the queue
    function<void(int, int)> markAsTwo = [&](int i, int j) {
      if (i < 0 || i == n || j < 0 || j == n) return;
      if (A[i][j] != 1) return;

      A[i][j] = -1;
      q.push({i, j});
      markAsTwo(i + 1, j);
      markAsTwo(i - 1, j);
      markAsTwo(i, j + 1);
      markAsTwo(i, j - 1);
    };

    [&]() {
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          if (A[i][j] == 1) {
            markAsTwo(i, j);
            return;
          }
    }();
    
    // expand by BFS
    while (!q.empty()) {
      for (int size = q.size(); size > 0; --size) {
        const auto [i, j] = q.front(); q.pop();
        for (int k = 0; k < 4; ++k) {
          const int x = i + dirs[k][0];
          const int y = j + dirs[k][1];
          if (x < 0 || x == n || y < 0 || y == n) continue;
          if (A[x][y] == -1) continue;
          if (A[x][y] == 1) return ans;
          A[x][y] = -1;
          q.push({x, y});
        }
      }
      ++ans;
    }

    throw;
  }
};

// Union-Find
grid = [[0,1,0],[0,0,0],[0,0,1]]
idx  =   0 1 2    3 4 5  6 7 8
nodes[i] = 0 0 -1 0 0 0 0 0 0 -1 
edges = (1,2) --> (2,3) --> 
        (1,4) --> (4,5) --> (2,5) --> 
        (5,6) --> (3,6) --> (4,7) --> (7,8) --> (5,8) --> (8,9) --> 
        (6,9)

merge_list = (1,2) --> (2,3) --> (2,5) --> (8,9) --> (6,9)
after mergeing remaining edges :
edges = (1,4) --> (4,5) --> (5,6) --> (3,6) --> 
(4,7) --> (7,8) --> (5,8)


int uf_find(int i, vector<int>& nodes) {
  if (nodes[i] <= 0) return i;
  else return nodes[i] = uf_find(nodes[i], nodes);
}
int uf_union(int i, int j, vector<int>& nodes) {
  auto pi = uf_find(i, nodes), pj = uf_find(j, nodes);
  if (pi == pj) return 0;
  if (nodes[pi] > nodes[pj]) swap(pi, pj);
  nodes[pi] += min(-1, nodes[pj]);
  nodes[pj] = pi;
  return -nodes[pi];
}
int shortestBridge(vector<vector<int>> &A) {
  int sz = A.size();
  vector<int> nodes(sz * sz + 1);
  list<pair<int, int>> edges;
  for (auto i = 0; i < sz; ++i)
    for (auto j = 0; j < sz; ++j) {
      auto idx = i * sz + j + 1; //idx=i*col+j  
      if (A[i][j]) nodes[idx] = -1; //visited
      if (j > 0) {
        // same row but previous column 1 then add to nodes 
        if (A[i][j] && A[i][j - 1]) uf_union(idx - 1, idx, nodes); 
        else edges.push_back({ idx - 1, idx });
      }
      if (i > 0) {
        // same column but previous row 1 then add to nodes
        if (A[i][j] && A[i - 1][j]) uf_union(idx - sz, idx, nodes);
        else edges.push_back({ idx - sz, idx });
      }
    }

  for (auto step = 1; ; ++step) {
    vector<pair<int, int>> merge_list;
    for (auto it = edges.begin(); it != edges.end(); ) {
      if (nodes[it->first] == 0 && nodes[it->second] == 0) ++it;
      else {
        if (nodes[it->first] != 0 && nodes[it->second] != 0) {
          if (uf_find(it->first, nodes) != uf_find(it->second, nodes)) return (step - 1) * 2;
        }
        merge_list.push_back({ it->first, it->second });
        edges.erase(it++);
      }
    }
    //if we can't merge further i.e no of group is 2, we return steps  
    //here calculation is done for 2 groups
    for (auto p : merge_list) {
      if (nodes[p.first] != 0 && nodes[p.second] != 0) {
        if (uf_find(p.first, nodes) != uf_find(p.second, nodes)) return step * 2 - 1;
      }
      uf_union(p.first, p.second, nodes);
    }
  }
}