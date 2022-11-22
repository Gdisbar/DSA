64. Minimum Path Sum
=======================
// Given a m x n grid filled with non-negative numbers, find a path from top 
// left to bottom right, which minimizes the sum of all numbers along its path.

// Note: You can only move either down or right at any point in time.

 								// 1 3 1
 								// 1 5 1
 								// 4 2 1

// Example 1:

// Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
// Output: 7
// Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

// Example 2:

// Input: grid = [[1,2,3],[4,5,6]]
// Output: 12


// Memoization

class Solution {
private:
int minSumPathUtil(int i, int j,vector<vector<int>> &grid,vector<vector<int>> &dp)
{
  if(i==0 && j == 0) return grid[0][0];
  if(i<0 || j<0) return 1e9;
  if(dp[i][j]!=-1) return dp[i][j];
    
  int up = grid[i][j]+minSumPathUtil(i-1,j,grid,dp);
  int left = grid[i][j]+minSumPathUtil(i,j-1,grid,dp);
  
  return dp[i][j] = min(up,left);
  
}
public:
    int minPathSum(vector<vector<int>>& grid) {
        int n=grid.size(),m=grid[0].size();
        vector<vector<int>> dp(n,vector<int>(m,-1));
        return minSumPathUtil(n-grid,dp);;
    
    }
};

// Tabulation

    int minPathSum(vector<vector<int>>& grid) {
        int m=grid.size(),n=grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));
        for(int i=0; i<m ; i++){
            for(int j=0; j<n; j++){
                if(i==0 && j==0) dp[i][j] = grid[i][j];
                else{

                    int up = grid[i][j];
                    if(i>0) up += dp[i-1][j];
                    else up += 1e9;

                    int left = grid[i][j];
                    if(j>0) left+=dp[i][j-1];
                    else left += 1e9;

                    dp[i][j] = min(up,left);
                }
            }
        }

        return dp[m-1][n-1];
    
    }

// Tabulation - Optimized

int minPathSum(vector<vector<int>>& grid) {
        int r=grid.size(),c=grid[0].size();
        //vector<vector<int>> dp(r,vector<int>(c,0));
        vector<int> prev(c,0);
        for(int i=0; i<r ; i++){
            vector<int> cur(c,0);
            for(int j=0; j<c; j++){
                if(i==0 && j==0) cur[j] = grid[i][j];
                else{

                    int up = grid[i][j];
                    if(i>0) up += prev[j];
                    else up += 1e9;

                    int left = grid[i][j];
                    if(j>0) left+=cur[j-1];
                    else left += 1e9;

                    cur[j] = min(up,left);
                }
            }
            prev=cur;
        }

        return prev[c-1];
    
    }