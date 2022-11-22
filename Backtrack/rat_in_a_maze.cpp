Rat in a Maze
=============================
// Consider a rat placed at (0, 0) in a square matrix of order N * N. 
// It has to reach the destination at (N – 1, N – 1). 
// Find all possible paths that the rat can take to reach from source to 
// destination. The directions in which the rat can move are ‘U'(up), 
// ‘D'(down), ‘L’ (left), ‘R’ (right). Value 0 at a cell in the matrix 
// represents that it is blocked and the rat cannot move to it while 
// value 1 at a cell in the matrix represents that rat can travel 
// through it.

// Note: In a path, no cell can be visited more than one time.

// Print the answer in lexicographical(sorted) order

// Examples:

// Example 1:

// Input:
// N = 4
// m[][] = {{1, 0, 0, 0},
//         {1, 1, 0, 1}, 
//         {1, 1, 0, 0},
//         {0, 1, 1, 1}}

// Output: DDRDRR DRDDRR

// Explanation:
//            f(0,0,"") D|L|R|U
//               |
//            f(1,0,"D") --------------------f(1,1,"DR")
//               | |                                   |
//             f(2,0"DD")                     f(2,1,"DRD")
//               | |                                   |
//             f(2,1,"DDR")----f(1,1,"DDRU")           |
//               |  |                                  |
//             f(3,1,"DDRD")                   f(3,1,"DRDD")
//               |   |                                 |
//             f(3,2,"DDRDR")                  f(3,2,"DRDDR")
//               |   |                                 |
//              f(3,3,"DDRDRR")                 f(3,3,"DRDDRR")



// For  “DDRDRR” :                    visited:

//        00 -  -  -                         00 -  -  -                    
//        10 -  -  -                         10 -  -  -                             
//        20 21 -  -                         20 21 -  -
//        -  31 32 33                        -  31 32 -



// The rat can reach the destination at (3, 3) from (0, 0) by two paths - 
// DRDDRR and DDRDRR, when printed in sorted order we get DDRDRR DRDDRR.

// Example 2:

// Input: N = 2
//        m[][] = {{1, 0},
//                 {1, 0}}

// Output:
//  No path exists and the destination cell is blocked.


// TC : 4^(m*n) , SC : m*n


class Solution {

//    | D  | L  | R  | U  | 
//    ---------------------
//    | +1 | +0 | +0 | -1 | <--- di
//    ---------------------
//    | +0 | -1 | +1 | +0 | <--- dj

int di[] = {+1,0,0,-1};
int dj[] = {0,-1,1,0};


oid solve(int i, int j, vector <vector<int>> &a, int n, 
       vector <string> &ans, string move,vector<vector<int>> &vis) {

    if (i == n - 1 && j == n - 1) {
      ans.push_back(move);
      return;
    }
    string dir = "DLRU";
    for (int ind = 0; ind < 4; ind++) {
      int nexti = i + di[ind];
      int nextj = j + dj[ind];
      if (nexti >= 0 && nextj >= 0 && nexti < n && nextj < n && !vis[nexti][nextj] 
                      && a[nexti][nextj] == 1) {
        vis[i][j] = 1;
        solve(nexti, nextj, a, n, ans, move + dir[ind], vis);
        vis[i][j] = 0;
      }
    }

  }
  // void solve(int i, int j, vector < vector < int >> & a, int n, vector < string > & ans, string move,
  //   vector < vector < int >> & vis) {
  //   if (i == n - 1 && j == n - 1) {
  //     ans.push_back(move);
  //     return;
  //   }

  //   // downward
  //   if (i + 1 < n && !vis[i + 1][j] && a[i + 1][j] == 1) {
  //     vis[i][j] = 1;
  //     solve(i + 1, j, a, n, ans, move + 'D', vis);
  //     vis[i][j] = 0;
  //   }

  //   // left
  //   if (j - 1 >= 0 && !vis[i][j - 1] && a[i][j - 1] == 1) {
  //     vis[i][j] = 1;
  //     solve(i, j - 1, a, n, ans, move + 'L', vis);
  //     vis[i][j] = 0;
  //   }

  //   // right 
  //   if (j + 1 < n && !vis[i][j + 1] && a[i][j + 1] == 1) {
  //     vis[i][j] = 1;
  //     solve(i, j + 1, a, n, ans, move + 'R', vis);
  //     vis[i][j] = 0;
  //   }

  //   // upward
  //   if (i - 1 >= 0 && !vis[i - 1][j] && a[i - 1][j] == 1) {
  //     vis[i][j] = 1;
  //     solve(i - 1, j, a, n, ans, move + 'U', vis);
  //     vis[i][j] = 0;
  //   }

  // }
  public:
    vector < string > findPath(vector<vector<int >> &mat, int n) {
      vector < string > ans;
      vector < vector < int >> vis(n, vector < int > (n, 0));
      if (mat[0][0] == 1) solve(0, 0, mat, n, ans, "", vis);
      return ans;
    }
};

// Solution obj;
// vector < string > result = obj.findPath(m, n);