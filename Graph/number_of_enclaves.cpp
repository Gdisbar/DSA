1020. Number of Enclaves
=============================
// You are given an m x n binary matrix grid, where 0 represents a sea cell 
// and 1 represents a land cell.

// A move consists of walking from one land cell to another adjacent 
// (4-directionally) land cell or walking off the boundary of the grid.

// Return the number of land cells in grid for which we cannot walk off 
// the boundary of the grid in any number of moves.

//  							[0,0,0,0]
//  							[1,0,1,0]
//  							[0,1,1,0]
//  							[0,0,0,0]

// Example 1:

// Input: grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
// Output: 3
// Explanation: There are three 1s that are enclosed by 0s, and one 1 that is 
// not enclosed because its on the boundary.
// 							[0,1,1,0]
// 							[0,0,1,0]
// 							[0,0,1,0]
// 							[0,0,0,0]
// Example 2:

// Input: grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
// Output: 0
// Explanation: All 1s are either on the boundary or can reach the boundary.

//WA

class Solution {
private:
    
	vector<int> dx={0,-1,1,0};
	vector<int> dy={-1,0,0,1};
    // int r,c;
    // bool isvalid(int x,int y){
    //     return x>=0&&x<r&&y>=0&&y<c;
    // }
public:
    int numEnclaves(vector<vector<int>>& grid) {

        int r = grid.size(),c=grid[0].size();
        int ones=0;
        vector<vector<bool>> dp(r,vector<bool>(c,false));
        for(int i = 0; i < r; ++i){
        	if(grid[i][0]==1) dp[i][0]=true;
        	if(grid[i][c-1]==1) dp[i][c-1]=true;
        }
        for(int j = 0; j < c; ++j){
        	if(grid[0][j]==1) dp[0][j]=true;
        	if(grid[r-1][j]==1) dp[r-1][j]=true;
        }
        int can=0;
        for(int i = 1; i < r-1; ++i){
        	for(int j = 1; j < c-1; ++j){
        		if(grid[i][j]){
                    ones++;
                    for(int k = 0; k < 4; ++k){
                        int x = i + dx[k];
                        int y = j + dy[k];
                        if(dp[x][y]){
                            dp[i][j]=true;
                            //break;
                        }
                    }
                    if(dp[i][j]) can++;
        			
        		}
        	}
        }
        cout<<ones<<" "<<can<<endl;
        return ones-can;

    }
};

// We flood-fill the land (change 1 to 0) from the boundary of the grid. 
// Then, we count the remaining land.


void dfs(vector<vector<int>>& A, int i, int j) {
  if (i < 0 || j < 0 || i == A.size() || j == A[i].size() || A[i][j] != 1) return;
  A[i][j] = 0;
  dfs(A, i + 1, j), dfs(A, i - 1, j), dfs(A, i, j + 1), dfs(A, i, j - 1);
}
int numEnclaves(vector<vector<int>>& A) {
  for (auto i = 0; i < A.size(); ++i)
    for (auto j = 0; j < A[0].size(); ++j) 
      if (i * j == 0 || i == A.size() - 1 || j == A[i].size() - 1)  //start from boundary
      	dfs(A, i, j);

  return accumulate(begin(A), end(A), 0, [](int s, vector<int> &r)
    { return s + accumulate(begin(r), end(r), 0); });
}

// BFS

int numEnclaves(vector<vector<int>>& A, int res = 0) {
  queue<pair<int, int>> q;
  for (auto i = 0; i < A.size(); ++i)
    for (auto j = 0; j < A[0].size(); ++j) {
      res += A[i][j];
      if (i * j == 0 || i == A.size() - 1 || j == A[i].size() - 1) 
      	q.push({ i, j });
    }
  while (!q.empty()) {
    auto x = q.front().first, y = q.front().second; q.pop();
    if (x < 0 || y < 0 || x == A.size() || y == A[x].size() || A[x][y] != 1) continue;
    A[x][y] = 0;
    --res;
    q.push({ x + 1, y }), q.push({ x - 1, y }), q.push({ x, y + 1 }), q.push({ x, y - 1 });
  }
  return res;
}