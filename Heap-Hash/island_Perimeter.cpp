463. Island Perimeter
==========================
// You are given row x col grid representing a map where grid[i][j] = 1 
// represents land and grid[i][j] = 0 represents water.

// Grid cells are connected horizontally/vertically (not diagonally). 
// The grid is completely surrounded by water, and there is exactly one 
// island (i.e., one or more connected land cells).

// The island doesn't have "lakes", meaning the water inside isn't connected 
// to the water around the island. One cell is a square with side length 1. 
// The grid is rectangular, width and height don't exceed 100. Determine the 
// perimeter of the island.

 

// Example 1:

// Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
// Output: 16
// Explanation: The perimeter is the 16 yellow stripes in the image above.

// Example 2:

// Input: grid = [[1]]
// Output: 4

// Example 3:

// Input: grid = [[1,0]]
// Output: 4

// class Solution {
// private:
//     vector<vector<bool>> vis;
//     int side=0,cell=0;
//     bool isvalid(vector<vector<int>>& grid,int i,int j){
//         if(i<0||i>grid.size()-1||j<0||j>grid[0].size()-1) return false;
//         if(grid[i][j]==0||vis[i][j]==true) return false;
//         return true;
//     }
//     void dfs(vector<vector<int>>& grid,int i,int j){
//         vis[i][j]=true;
//         cell++;
//         if(isvalid(grid,i-1,j)){
//             dfs(grid,i-1,j);
//             side++;
//         }
//         if(isvalid(grid,i,j-1)){
//             dfs(grid,i,j-1);
//             side++;
//         }
//         if(isvalid(grid,i+1,j)){
//             dfs(grid,i+1,j);
//             side++;
//         }
//         if(isvalid(grid,i,j+1)){
//             dfs(grid,i,j+1);
//             side++;
//         }
//     }
// public:
//     int islandPerimeter(vector<vector<int>>& grid) {
//         int r = grid.size(),c=grid[0].size(),ans=0;
//         vis.resize(r,vector<bool>(c,false));
//         for(int i = 0;i<r;++i){
//             for(int j=0;j<c;++j){
//                 if(!vis[i][j]&&grid[i][j]){
//                     side=0;
//                     cell=0;
//                     dfs(grid,i,j);
//                     cout<<side<<" "<<cell;
//                     ans+=cell*4-side*2;
//                 }
//             }
//         }
//         return ans;
//     }
// };

// missed out part

// for all grid(i,j)==1
// if valid(i,j) and grid(i,j) or !valid(i,j) --> increase cnt
// else if valid(i,j) and !vis(i,j) --> call dfs


int islandPerimeter(vector<vector<int>>& grid) {
        int r = grid.size(),c=grid[0].size();
        int cell= 0,side= 0;
        for(int i = 0;i<r;++i){
            for(int j=0;j<c;++j){
                if(grid[i][j]==1){
                    cell++;         // count no of island
                    if(i<r-1&&grid[i+1][j]==1) 
                        side++; // count if it has down neighbour
                    if(j<c-1&&grid[i][j+1]==1) 
                        side++; //count if it has right neighbour
                }
            }
        }
        return cell*4-side*2;
    }

//recursion
public int islandPerimeter(int[][] grid) {
    int m = grid.length;
    if(m == 0) return 0;
    int n = grid[0].length;
    int[][] dir = {{0,1},{1,0},{-1,0},{0,-1}};
    
    for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
        if(grid[i][j] == 1){
          return helper(grid, dir, i, j);
        }
      }
    }
    return 0;
  }

  int helper(int[][] grid, int[][] dir, int i, int j){
    
    int m = grid.length, n = grid[0].length;
    grid[i][j] = -1;
    int count = 0;
    for(int[] d: dir){
      int x = i + d[0];
      int y = j + d[1];
      if(x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0){
        count++;
      } else {
        if(grid[x][y] == 1){
            count += helper(grid, dir, x, y);
        }  
      }
    }
    return count;
  }