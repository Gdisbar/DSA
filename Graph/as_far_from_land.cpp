1162. As Far from Land as Possible
======================================
// Given an n x n grid containing only values 0 and 1, where 0 represents water 
// and 1 represents land, find a water cell such that its distance to the nearest 
// land cell is maximized, and return the distance. If no land or water exists in 
// the grid, return -1.

// The distance used in this problem is the Manhattan distance: the distance between 
// two cells (x0, y0) and (x1, y1) is |x0 - x1| + |y0 - y1|.

 

// Example 1:

// Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
// Output: 2
// Explanation: The cell (1, 1) is as far as possible from all the land with 
// distance 2.

// Example 2:

// Input: grid = [[1,0,0],[0,0,0],[0,0,0]]
// Output: 4
// Explanation: The cell (2, 2) is as far as possible from all the land with 
// distance 4.

// Complexity Analysis

// Runtime: O(m * n * n), where m is the number of land cells.
// Memory: O(n * n) for the recursion.
//TLE 
class Solution {
private:
	  
	void dfs(vector<vector<int>>& grid,int x,int y,int dist){
        if(x<0||y<0||x==grid.size()||y==grid[0].size()||
        	(grid[x][y]!=0 && grid[x][y] <= dist)) return;
        grid[x][y] = dist;
        dfs(grid,x+1,y,dist+1);
        dfs(grid,x-1,y,dist+1);
        dfs(grid,x,y+1,dist+1);
        dfs(grid,x,y-1,dist+1);
    }
public:
    int maxDistance(vector<vector<int>>& grid) {
        int r=grid.size(),c=grid[0].size();
        for(int i=0;i<r;++i){
        	for(int j=0;j<c;++j){
        		if(grid[i][j]==1){
        			grid[i][j]=0;
        			dfs(grid,i,j,1);
        		}
        	}
        }
        int mx=-1;
        for(int i=0;i<r;++i){
        	for(int j=0;j<c;++j){
        		if(grid[i][j]>1)
        			mx=max(mx,grid[i][j]-1);
        	}
        }
        return mx;
    }
};


// Complexity Analysis

// Runtime: O(n * n). We process an individual cell only once (or twice).
// Memory: O(n) for the queue.

int maxDistance(vector<vector<int>>& g, int steps = 0) {
  queue<pair<int, int>> q, q1;
  for (auto i = 0; i < g.size(); ++i)
    for (auto j = 0; j < g[i].size(); ++j)
      if (g[i][j] == 1)
        q.push({ i - 1, j }), q.push({ i + 1, j }), q.push({ i, j - 1 }), q.push({ i, j + 1 });
  while (!q.empty()) {
    ++steps;
    while (!q.empty()) {
      int i = q.front().first, j = q.front().second;
      q.pop();
      if (i >= 0 && j >= 0 && i < g.size() && j < g[i].size() && g[i][j] == 0) {
        g[i][j] = steps;
        q1.push({ i - 1, j }), q1.push({ i + 1, j }), q1.push({ i, j - 1 }), q1.push({ i, j + 1 });
      }
    }
    swap(q1, q);
  }
  return steps == 1 ? -1 : steps - 1;
}

// DP -> because O(n*n) is less than O(n*n + C), 
//in bfs solution we are inserting pair of integer and removing this.

// The idea is completly the same as 542. 01 Matrix.
// We maintain a dp table, the entries in the dp table represent the distance 
// to the nearest '1' + 1, why +1? Will explain this in a second.
// We traverse the grid 2 times, first from left up -> bottom right, 
// second from bottom right -> left up.
// In the first loop, we update the minimum distance to reach a '1' from the 
// current position either keep going left or going upward. Here's a small trick, 
// i pick 201 as the max value, cuz per the problem description, the # of rows 
// won't exceed 100, so the length of longest path in the matrix will not exceed 200.
// Say, a matrix A, after the first loop, it will become

// [1, 0, 0]      [1, 2, 3]
// [0, 0, 0]  ->  [2, 3, 4]
// [0, 0, 1]      [3, 4, 1]

// please note that this is not the real distance

// In the second loop, we go from bottom right to up left to update the 
// min distance from another side. At the end, please not that res is not 
// the value we want, if there's no '1's in the matrix, all the entry will 
// be set to 201 in such a case, we should return -1 instead of 201; if there 
// are '1's in the matrix, as mentioned at the begining, res is the 
// maximum distance + 1, so we need res-1.

// [1, 2, 3]    [1, 2, 3]  real distance [0, 1, 2]
// [2, 3, 4] -> [2, 3, 2]        ->      [1, 2, 1]
// [3, 4, 1]    [3, 2, 1]        -1      [2, 1, 0]

// the maximum value in the table is 3, this means the answer is 3 - 1 = 2.
// time/space: O(nm)/O(1)


class Solution {

public:
    int maxDistance(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size(), res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) continue;
                grid[i][j] = 201; //size won't exceed 100.
                if (i > 0) grid[i][j] = min(grid[i][j], grid[i-1][j] + 1);
                if (j > 0) grid[i][j] = min(grid[i][j], grid[i][j-1] + 1);
            }
        }
        
        for (int i = n-1; i > -1; i--) {
            for (int j = m-1; j > -1; j--) {
                if (grid[i][j] == 1) continue;
                if (i < n-1) grid[i][j] = min(grid[i][j], grid[i+1][j] + 1);
                if (j < m-1) grid[i][j] = min(grid[i][j], grid[i][j+1] + 1);
                res = max(res, grid[i][j]); //update the maximum
            }
        }
        
        return res == 201 ? -1 : res - 1;
    }
    
};