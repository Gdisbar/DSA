778. Swim in Rising Water
==========================
// You are given an n x n integer matrix grid where each value grid[i][j] 
// represents the elevation at that point (i, j).

// The rain starts to fall. At time t, the depth of the water everywhere is t. 
// You can swim from a square to another 4-directionally adjacent square if and 
// only if the elevation of both squares individually are at most t. You can swim 
// infinite distances in zero time. Of course, you must stay within the boundaries 
// of the grid during your swim.

// Return the least time until you can reach the bottom right square (n - 1, n - 1) 
// if you start at the top left square (0, 0).

 

// Example 1:

// Input: grid = [[0,2],[1,3]]
// Output: 3
// Explanation:
// At time 0, you are in grid location (0, 0).
// You cannot go anywhere else because 4-directionally adjacent neighbors have 
// a higher elevation than t = 0.
// You cannot reach point (1, 1) until time 3.
// When the depth of water is 3, we can swim anywhere inside the grid.

// Example 2:

// Input: grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],
//     [10,9,8,7,6]]
// Output: 16
// Explanation: The final route is shown.
// We need to wait until time 16 so that (0, 0) and (4, 4) are connected.




class Solution {
    struct T {
        int t, x, y;
        T(int a, int b, int c) : t (a), x (b), y (c){}
        bool operator< (const T &d) const {return t > d.t;} // for sorting
    };
    vector<vector<int>> dir={{0, 1}, {0, -1}, {1, 0}, { -1, 0}};
public:
    int swimInWater(vector<vector<int>>& grid) {
        int n=grid.size(),ans=0;
        priority_queue<T> pq;
        vector<vector<bool>> vis(n,vector<bool>(n,false));
        pq.push(T(grid[0][0], 0, 0));
        vis[0][0]=true;
        while(1){
            auto [t,x,y]=pq.top();
            pq.pop();
            ans=max(ans,t);
            if(x==n-1&&y==n-1) break;
            for(int i=0;i<dir.size();++i){
                int dx=x+dir[i][0];
                int dy=y+dir[i][1];
                //if(dx==n-1&&dy==n-1) break;
                if(dx>=0&&dx<n&&dy>=0&&dy<n&&vis[dx][dy]==false){
                    pq.push(T(grid[dx][dy],dx,dy));
                    vis[dx][dy]=true;
                }
            }
        }
        return ans;
    }
};


 def swimInWater(self, grid):
        N, pq, seen, res = len(grid), [(grid[0][0], 0, 0)], set([(0, 0)]), 0
        while True:
            T, x, y = heapq.heappop(pq)
            res = max(res, T)
            if x == y == N - 1:
                return res
            for i, j in [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]:
                if 0 <= i < N and 0 <= j < N and (i, j) not in seen:
                    seen.add((i, j))
                    heapq.heappush(pq, (grid[i][j], i, j))

// using priority queue
// time complexity: O(n^2*logn)

//     pq contains at most n^2 elements, pop time complexity each time is is 
//     O(logn^2) = O(2*logn)
//     At most we will pop n^2 times

// O(n^2*2*logn) = O(n^2*logn)

  
int swimInWater(vector<vector<int>>& grid) {
      int t = 0, n = grid.size();
      int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
      // min pq, based on grid value
      // we will store {value, x coordinate, y coordinate}
      priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;

      // push (0,0) start point
      pq.push({grid[0][0], 0, 0});
      grid[0][0] = -1;

      while(pq.size()){

        // keep on doing bfs, until t >= value 
        // we will do bfs from min value to max, becase min time will be first
        // that's why we used min pq
        while(pq.size() && pq.top()[0] <= t){
          auto curr = pq.top(); pq.pop();

          int x = curr[1], y = curr[2];

          // if reached at end, return time
          if(x == n-1 && y == n-1) return t;

          // add next positions which are not visited into pq
          for(auto &dir: dirs){
            int nx = x+dir[0], ny = y+dir[1];
            if(nx>=0 && ny>=0 && nx<n && ny<n && grid[nx][ny] != -1){
              pq.push({grid[nx][ny], nx, ny});
              grid[nx][ny] = -1;
            }
          }
        } 

        // we have done all bfs in this time, we need to increment t
        t++;
      }
       return -1;
}