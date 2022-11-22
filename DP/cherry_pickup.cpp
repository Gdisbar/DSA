741. Cherry Pickup
====================
// You are given an n x n grid representing a field of cherries, each cell is 
// one of three possible integers.

//     0 means the cell is empty, so you can pass through,
//     1 means the cell contains a cherry that you can pick up and pass through, or
//     -1 means the cell contains a thorn that blocks your way.

// Return the maximum number of cherries you can collect by following the rules below:

//     Starting at the position (0, 0) and reaching (n - 1, n - 1) by moving right or 
//     down through valid path cells (cells with value 0 or 1).
//     After reaching (n - 1, n - 1), returning to (0, 0) by moving left or up through 
//     valid path cells.
//     When passing through a path cell containing a cherry, you pick it up, and the 
//     cell becomes an empty cell 0.
//     If there is no valid path between (0, 0) and (n - 1, n - 1), then no cherries 
//     can be collected.

 


// Example 1:

// Input: grid = [[0,1,-1],[1,0,-1],[1,1,1]]
// Output: 5
// Explanation: The player started at (0, 0) and went down, down, right right to 
// reach (2, 2).
// 4 cherries were picked up during this single trip, and the matrix becomes 
// [[0,1,-1],[0,0,-1],[0,0,0]].
// Then, the player went left, up, up, left to return home, picking up one more cherry.
// The total number of cherries picked up is 5, and this is the maximum possible.

// Example 2:

// Input: grid = [[1,1,-1],[1,-1,1],[-1,1,1]]
// Output: 0


//TLE 
class Solution {
private:
    int final_cherry=0;
    void dfs2(vector<vector<int>>& grid,int r,int c,int cherry){
    	if(r<0||r==grid.size()||c<0||c==grid[0].size()||grid[r][c]==-1){
    		return;
    	}
    	if(r==0&&c==0){
    		final_cherry=max(final_cherry,cherry);
    		return;
    	}
    	int cherry_count=grid[r][c];
    	grid[r][c]=0;
    	dfs2(grid,r,c-1,cherry+cherry_count);
    	dfs2(grid,r-1,c,cherry+cherry_count);
    	grid[r][c]=cherry_count;
    }
    void dfs1(vector<vector<int>>& grid,int r,int c,int cherry){
    	if(r<0||r==grid.size()||c<0||c==grid[0].size()||grid[r][c]==-1){
    		return;
    	}
    	if(r==grid.size()-1&&c==grid[0].size()-1){
    		dfs2(grid,r,c,cherry);
    	}
    	int cherry_count=grid[r][c];
    	grid[r][c]=0;
    	dfs1(grid,r,c+1,cherry+cherry_count);
    	dfs1(grid,r+1,c,cherry+cherry_count);
    	grid[r][c]=cherry_count;
    }
public:
    int cherryPickup(vector<vector<int>>& grid) {
        if(grid.size()== 1 && grid[0][0]== 1)  return 1;
        dfs1(grid,0,0,0);
        return final_cherry;
    }
};

// Approach - 2 , TLE - rather than returning from destination if two person 
//reach destination that will be same
// Now Wrong answer for [[1]]
int dfs(vector<vector<int>>& grid,int r1,int c1,int r2,int c2){
        if(r1>=grid.size()||r2>=grid.size()||c1>=grid[0].size()||
            c2>=grid[0].size()||grid[r1][c1]==-1||grid[r2][c2]==-1){
            return 0;
        }
        if(r1==grid.size()-1&&c1==grid[0].size()-1){  //both reached destination
            return grid[r1][c1];
        }
        
        int cherry = 0;
        if(r1==r2&&c1==c2){ //both are on same grid
            cherry+=grid[r1][c1];
        }
        else{         //both are on different grid
            cherry+=grid[r1][c1]+grid[r2][c2];
        }
        int v1=dfs(grid,r1,c1+1,r2,c2+1,dp); //h,h
        int v2=dfs(grid,r1+1,c1,r2,c2+1,dp);//v,h
        int v3=dfs(grid,r1,c1+1,r2+1,c2,dp);//h,v
        int v4=dfs(grid,r1+1,c1,r2+1,c2,dp);//v,v
        cherry+=max(max(v1,v2),max(v3,v4));
        return cherry;
    }




//3-D dp - not working
int cherryPickup(vector<vector<int>>& grid){
    int r=grid.size(),c=grid[0].size();
    vector<vector<vector<int>>> dp(r,vector<vector<int>>(c,vector<int>(c,0)));

    for(int j1=0;j1<c;++j1){
        for(int j2=0;j2<c;++j2){
            if(j1==j2)
                dp[r-1][j1][j2]=grid[r-1][j1];
            else
                dp[r-1][j1][j2]=grid[r-1][j1]+grid[r-1][j2];
        }
    }

    for(int i=r-2;i>=0;--i){
        for(int j1=0;j1<c;++j1){
            for(int j2=0;j2<c;++j2){
                int mx = -1e8;
                for(int dj1=-1;dj1<=1;dj1++){
                    for(int dj2=-1;dj2<=1;dj2++){
                        int value=0;
                        if(j1==j2)
                            value = grid[i][j1];
                        else
                            value = grid[i][j1]+grid[i][j2];
                        if(j1+dj1 >=0 && j1+dj1 < c && j2+dj2>=0 && j2+dj2 < c)
                            value+=dp[i+1][j1+dj1][j2+dj2];
                        else 
                            value+=-1e8;
                        mx=max(mx,value);
                    }
                }
                dp[i][j1][j2]=value;
            }
        }
    }

    return dp[0][0][c-1];
}


//https://leetcode.com/problems/cherry-pickup/discuss/109903/Step-by-step-guidance-of-the-O(N3)-time-and-O(N2)-space-solution



int cherryPickup(vector<vector<int>>& grid) {
        int N = grid.size();
        vector<vector<int>> dp(N, vector<int>(N, -1));
        // dp holds maximum # of cherries two k-length paths can pickup.
        // The two k-length paths arrive at (i, k - i) and (j, k - j), 
        // respectively.
        dp[0][0] = grid[0][0]; // length k = 0
        // maxK: number of steps from (0, 0) to (n-1, n-1).
        for (int k = 1, maxK = 2*N-2; k <= maxK; ++k) { 
            // one path of length k arrive at (i, k - i) 
            for (int i = min(N-1, k); i >= 0; --i) {
                if (k - i >= N) continue;
                // another path of length k arrive at (j, k - j)
                for (int j = min(N-1, k); j >= 0; --j) {
                    if (k - j >= N || grid[i][k - i] < 0 || grid[j][k - j] < 0) {
                        // keep away from thorns
                        dp[i][j] = -1;
                        continue;
                    }
                    // # of cherries picked up by the two (k-1)-length paths.
                    int cherris = dp[i][j]; 
                    // See the figure below for an intuitive understanding
                    if (i > 0)  cherris = max(cherris, dp[i-1][j]);
                    if (j > 0)  cherris = max(cherris, dp[i][j-1]);
                    if (i > 0 && j > 0)  cherris = max(cherris, dp[i-1][j-1]);
                    // No viable way to arrive at (i, k - i)-(j, k-j).
                    if (cherris < 0)    continue;
                    // Pickup cherries at (i, k - i) and (j, k -j ) if i != j.
                    // Otherwise, pickup (i, k-i). 
                    dp[i][j] =  cherris + grid[i][k-i];
                    if (i != j) dp[i][j] += grid[j][k-j];
                }
            }
        }
        return max(0, dp[N-1][N-1]);
    }

// Memoization

// def cherryPickup(self, grid):
//         N = len(grid)
//         lookup = {}
        
//         def solve(x1, y1, x2, y2):
//             # check if we reached bottom right corner
//             if x1 == N-1 and y1 == N-1: 
//                 return grid[x1][y1] if grid[x1][y1] != -1 else float("-inf")
            
//             # out of the grid and thorn check
//             if x1 == N or y1 == N or x2 == N or y2 == N or grid[x1][y1] == -1 or grid[x2][y2] == -1: 
//                 return float("-inf")
            
//             # memorization check
//             lookup_key = (x1, y1, x2, y2)
//             if lookup_key in lookup: return lookup[lookup_key]
            
//             # pick your cherries
//             if x1 == x2 and y1 == y2:
//                 cherries = grid[x1][y1]
//             else:
//                 cherries = grid[x1][y1] + grid[x2][y2]
                
//             res = cherries + max(
//                 solve(x1 + 1, y1, x2 + 1, y2),  # right, right
//                 solve(x1, y1 + 1, x2, y2 + 1),  # down, down
//                 solve(x1 + 1, y1, x2, y2 + 1),  # right, down
//                 solve(x1, y1 + 1, x2 + 1, y2), # down, right
//             )
            
//             lookup[lookup_key] = res
//             return res
        
//         res = solve(0, 0, 0, 0)
//         return res if res > 0 else 0

