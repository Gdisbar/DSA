934. Shortest Bridge
=====================
You are given an n x n binary matrix grid where 1 represents land and 0 
represents water.

An island is a 4-directionally connected group of 1's not connected to 
any other 1's. There are exactly two islands in grid.

You may change 0's to 1's to connect the two islands to form one island.

Return the smallest number of 0's you must flip to connect the two islands.

 

Example 1:

Input: grid = [[0,1],[1,0]]
Output: 1

Example 2:

Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2

Example 3:

Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1


// Not working

class Solution {
private:
   
    vector<vector<int>> dir = {{-1,0},{0,1},{1,0},{0,-1}};
    vector<pair<int,int>> v;
     bool isvalid(vector<vector<int>>& grid,vector<vector<bool>>& vis,int x,int y){
        if(x< 0 ||y< 0||x >= grid.size() ||y >= grid[0].size()||vis[x][y])
            return false;
        return true;
    }
    void bfs(vector<vector<int>>& grid,vector<vector<bool>>& vis,int row,int col){
        queue<pair<int, int> > q;
        q.push({ row, col });
        vis[row][col] = true;
        
        while (!q.empty()) {
            pair<int, int> cell = q.front();
            int x = cell.first;
            int y = cell.second;
            q.pop();
            v.push_back({x,y});
            for (int i = 0; i < 4; i++) {

                int adjx = x + dir[i][0];
                int adjy = y + dir[i][1];

                if (isvalid(grid,vis,adjx,adjy)) {
                    q.push({ adjx, adjy });
                    vis[adjx][adjy] = true;
                }
            }
         }
    }
public:
    int shortestBridge(vector<vector<int>>& grid) {
        int r=grid.size(),c=grid[0].size();
        vector<vector<bool>> vis(r,vector<bool>(c,false));
        
        for(int i=0;i<r;++i){
            for(int j=0;j<c;++j){
                if(grid[i][j]==1){
                    bfs(grid,vis,i,j);
                    break;
                }
            }
        }
        int cnt=0;
        for(int i=0;i<v.size();++i){
            int x = v[i].first;
            int y = v[i].second;
            for(int j=0;j<4;++j){
                int adjx=x+dir[i][0];
                int adjy=y+dir[i][1];
                if(grid[adjx][adjy]==1&&vis[adjx][adjy]==false){
                    vis[adjx][adjy]=true;
                    break;
                }
                else if(grid[adjx][adjy]==0&&vis[adjx][adjy]==false){
                    cnt++;
                }
            }
        }
        return cnt;
    }
};

// Working
class Solution {
private:
    vector<vector<int>> dir = {{-1,0},{0,1},{1,0},{0,-1}};
    queue<pair<int,int> > q;
    void dfs(vector<vector<int>>& grid,int x,int y){
        if(x<0||y<0||x==grid.size()||y==grid[0].size()||grid[x][y]!=1) return;
        q.push({x,y});
        grid[x][y] = -1;
        dfs(grid,x+1,y);
        dfs(grid,x-1,y);
        dfs(grid,x,y+1);
        dfs(grid,x,y-1);
    }
public:
    int shortestBridge(vector<vector<int>>& grid) {
        int r=grid.size(),c=grid[0].size();
        bool flag=false;
        for(int i=0;i<r;++i){
            for(int j=0;j<c;++j){
                if(grid[i][j]==1){
                    dfs(grid,i,j);
                    flag=true;
                    break;
                }
            }
            if(flag) break;
        }
        int cnt=0;
          while (!q.empty()) {
            int sz = q.size();
            for(int i=0;i<sz;++i){
                pair<int,int> cell = q.front();
                int x = cell.first;
                int y = cell.second;
                q.pop();
                for (int i = 0; i < 4; i++) {
                    int dx = x + dir[i][0];
                    int dy = y + dir[i][1];
                    if (dx >= 0 && dy >= 0 && dx < r && dy < c && grid[dx][dy] != -1) {
                        q.push({dx,dy});
                        if(grid[dx][dy]==1) return cnt;
                        grid[dx][dy] = -1;
                    }
               }
            } 
              cnt++;
         }
        return cnt;
    }
};

//faster

class Solution {
private:
	int dir[5] = {0, 1, 0, -1, 0};
	//paint using color = 2 , using dfs
	void paint(vector<vector<int>>& A, int i, int j, vector<pair<int, int>> &q) {
	    if (min(i, j) >= 0 && max(i, j) < A.size() && A[i][j] == 1) {
	        A[i][j] = 2;
	        q.push_back({i, j});
	        for (int d = 0; d < 4; ++d)
	            paint(A, i + dir[d], j + dir[d + 1], q);
	    }
	}
public:
	int shortestBridge(vector<vector<int>>& A) {
	    vector<pair<int, int>> q;
	    for (int i = 0; q.size() == 0 && i < A.size(); ++i)
	        for (int j = 0; q.size() == 0 && j < A[0].size(); ++j)
	            paint(A, i, j, q);

	    while (!q.empty()) {
	        vector<pair<int, int>> q1;
	        for (auto [i, j] : q) {
	            for (int d = 0; d < 4; ++d) {
	                int x = i + dir[d], y = j + dir[d + 1];
	                if (min(x, y) >= 0 && max(x, y) < A.size()) {
	                    if (A[x][y] == 1)
	                        return A[i][j] - 2;
	                    //paint connected empty area of island (previously painted 2)
	                    if (A[x][y] == 0) {
	                        A[x][y] = A[i][j] + 1;
	                        q1.push_back({x, y});
	                    }
	                }
	            }
	        }
	        swap(q, q1);
	    }
	    return 0;
	}
};

// using UF
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
      auto idx = i * sz + j + 1;
      if (A[i][j]) nodes[idx] = -1;
      if (j > 0) {
        if (A[i][j] && A[i][j - 1]) uf_union(idx - 1, idx, nodes);
        else edges.push_back({ idx - 1, idx });
      }
      if (i > 0) {
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
    for (auto p : merge_list) {
      if (nodes[p.first] != 0 && nodes[p.second] != 0) {
        if (uf_find(p.first, nodes) != uf_find(p.second, nodes)) return step * 2 - 1;
      }
      uf_union(p.first, p.second, nodes);
    }
  }
}