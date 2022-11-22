51. N-Queens
================
// The n-queens puzzle is the problem of placing n queens on an n x n 
// chessboard such that no two queens attack each other.

// Given an integer n, return all distinct solutions to the n-queens puzzle. 
// You may return the answer in any order.

// Each solution contains a distinct board configuration of the 
// n-queens' placement, where 'Q' and '.' both indicate a queen and an 
// empty space, respectively.

 

// Example 1:
//                   -Q--      --Q-
//                   ---Q      Q---
//                   Q---      ---Q
//                   --Q-      -Q--
// Input: n = 4
// Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
// Explanation: There exist two distinct solutions to the 4-queens puzzle 
// as shown above

// Example 2:

// Input: n = 1
// Output: [["Q"]]

//TC : n!*n 

class Solution {
  public:
    bool isSafe1(int row, int col, vector < string > board, int n) {
                   
                 //   ^ 
                 //    \           (upper diagonal)
                 //     \
                 // <----- element (row check)
                 //      /          
                 //     /          (lower diagonal)
                     
      // check upper element
      int duprow = row;
      int dupcol = col;

      while (row >= 0 && col >= 0) { //upper diagonal check , TC : n
        if (board[row][col] == 'Q')
          return false;
        row--;
        col--;
      }

      col = dupcol;
      row = duprow;
      while (col >= 0) { //row check  , TC : n
        if (board[row][col] == 'Q')
          return false;
        col--;
      }

      row = duprow;
      col = dupcol;
      while (row < n && col >= 0) { //lower diagonal check , TC : n
        if (board[row][col] == 'Q')
          return false;
        row++;
        col--;
      }
      return true;
    }

  public:
    void solve(int col, vector<string>&board, vector<vector<string>>&ans,int n) {
      if (col == n) {
        ans.push_back(board);
        return;
      }
      for (int row = 0; row < n; row++) {
        if (isSafe1(row, col, board, n)) {
          board[row][col] = 'Q';
          solve(col + 1, board, ans, n);
          board[row][col] = '.';
        }
      }
    }

  public:
    vector < vector < string >> solveNQueens(int n) {
      vector < vector < string >> ans;
      vector < string > board(n);
      string s(n, '.');
      for (int i = 0; i < n; i++) {
        board[i] = s;
      }
      solve(0, board, ans, n);
      return ans;
    }
};

// for (int i = 0; i < ans.size(); i++) {
//     cout << "Arrangement " << i + 1 << "\n";
//     for (int j = 0; j < ans[0].size(); j++) {
//       cout << ans[i][j];
//       cout << endl;
//     }
//     cout << endl;

//Space Optimized , TC : same as above , SC : n

class Solution {
  public:
    void solve(int col, vector < string > & board, vector < vector < string >> & ans, 
                        vector < int > & leftrow, vector < int > & upperDiagonal, 
                        vector < int > & lowerDiagonal, int n) {
      if (col == n) {
        ans.push_back(board);
        return;
      }
      for (int row = 0; row < n; row++) {
        if (leftrow[row] == 0 && lowerDiagonal[row + col] == 0 
                          && upperDiagonal[n - 1 + col - row] == 0) {
          board[row][col] = 'Q';
          leftrow[row] = 1;
          lowerDiagonal[row + col] = 1;
          upperDiagonal[n - 1 + col - row] = 1;
          solve(col + 1, board, ans, leftrow, upperDiagonal, lowerDiagonal, n);
          board[row][col] = '.';
          leftrow[row] = 0;
          lowerDiagonal[row + col] = 0;
          upperDiagonal[n - 1 + col - row] = 0;
        }
      }
    }

  public:
    vector < vector < string >> solveNQueens(int n) {
      vector < vector < string >> ans;
      vector < string > board(n);
      string s(n, '.');
      for (int i = 0; i < n; i++) {
        board[i] = s;
      }
      vector < int > leftrow(n, 0), upperDiagonal(2 * n - 1, 0), lowerDiagonal(2 * n - 1, 0);
      solve(0, board, ans, leftrow, upperDiagonal, lowerDiagonal, n);
      return ans;
    }
};