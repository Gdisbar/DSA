994. Rotting Oranges
====================
// You are given an m x n grid where each cell can have one of three values:

//     0 representing an empty cell,
//     1 representing a fresh orange, or
//     2 representing a rotten orange.

// Every minute, any fresh orange that is 4-directionally adjacent to a rotten 
// orange becomes rotten.

// Return the minimum number of minutes that must elapse until no cell has a fresh 
// orange. If this is impossible, return -1.

 

// Example 1:

// Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
// Output: 4

// Example 2:

// Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
// Output: -1
// Explanation: The orange in the bottom left corner (row 2, column 0) is never 
// rotten, because rotting only happens 4-directionally.

// Example 3:

// Input: grid = [[0,2]]
// Output: 0
// Explanation: Since there are already no fresh oranges at minute 0, the answer 
// is just 0.


int orangesRotting(vector<vector<int>>& grid) {
       vector<vector<int>> dir={{0,1},{1,0},{0,-1},{-1,0}};
       //vector<int> dir={-1,0,1,0,-1};
        int n=grid.size();
        int m=grid[0].size();
        queue<pair<int,int>> q;
        int fresh=0; 
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++)
            {
                if(grid[i][j]==2)
                    q.push({i,j});
                if(grid[i][j]==1)
                    fresh++;
            }
        }
        //initialised to -1 since after each step we increment the time by 1 
        //and initially all rotten oranges started at 0.   
        int time=-1;
        while(!q.empty())
        {
            int sz=q.size();
            while(sz--){
                int rr=q.front().first;
                int rc=q.front().second;
                q.pop();
                for(int i=0;i<4;i++)
                {
                    int r=rr+dir[i][0];
                    int c=rc+dir[i][1];
                    if(r>=0 && r<n && c>=0 && c<m &&grid[r][c]==1)
                    {
                        grid[r][c]=2;
                        q.push({r,c});
                        fresh--; 
                    }
                    
                }
            }
         time++; 
        }
        // for(int i=0;i<grid.size();++i){
        //     for(int j=0;j<grid[i].size();++j){
        //         if(grid[i][j]==1){
        //             return -1;
        //         }
        //     }
        // }

        //if fresh>0 that means there are fresh oranges left
        if(fresh>0) return -1;
        //we initialised with -1, so if there were no oranges it'd take 0 mins.
        if(time==-1) return 0;
        return time;
  }
