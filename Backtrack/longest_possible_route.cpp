329. Longest Increasing Path in a Matrix
===========================================
// Given an m x n integers matrix, return the length of the longest 
// increasing path in matrix.

// From each cell, you can either move in four directions: left, right, up, 
// or down. You may not move diagonally or move outside the boundary 
// (i.e., wrap-around is not allowed).

 

// Example 1:
//                         9    9 4
//                         ^
//                         |
//                         6    6 8
//                         ^
//                         |
//                         2<---1 1


// Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
// Output: 4
// Explanation: The longest increasing path is [1, 2, 6, 9].

// Example 2:

// Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
// Output: 4
// Explanation: The longest increasing path is [3, 4, 5, 6]. 
// Moving diagonally is not allowed.

// Example 3:

// Input: matrix = [[1]]
// Output: 1


// Top-down 
//TC & SC : n*m

int moves[4][2] = { {-1,0},{1,0},{0,-1},{0,1} };
// dp[i][j] will store maximum path length starting from matrix[i][j]
int dp[200][200]{}; // constraints are small enough that we can just set them to MAX
int maxPath, n, m;
int longestIncreasingPath(vector<vector<int>>& matrix) {
    maxPath = 0, n = size(matrix), m = size(matrix[0]);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            maxPath = max(maxPath, solve(matrix, i, j, -1));            
    return maxPath;
}
// // recursive solver for each cell with dp for storing each calculated result
// int solve(vector<vector<int>>& mat, int i, int j){
//     if(dp[i][j]) return dp[i][j]; // return if result is already calculated
//     dp[i][j] = 1;  // minimum path from each cell is always atleast 1
//     // choosing each possible move available to us
//     for(int k = 0; k < 4; k++){ 
//         int new_i = i + moves[k][0], new_j = j + moves[k][1];
//         // bound checking as well as move to next cell only when it is greater in value
//         if(new_i < 0 || new_j < 0 || new_i >= n || new_j >= m || mat[new_i][new_j] <= mat[i][j]) continue;
//         // max( current optimal, select current + optimal solution after moves[k] from current cell
//         dp[i][j] = max(dp[i][j], 1 + solve(mat, new_i, new_j));
//     }         
//     return dp[i][j];
// }

int solve(vector<vector<int>>& mat, int i, int j, int prev){
    if(i < 0 || j < 0 || i >= n || j >= m || mat[i][j] <= prev) return 0;
    if(dp[i][j]) return dp[i][j];
    return dp[i][j] = 1 + max({ solve(mat, i + 1, j, mat[i][j]),
                                solve(mat, i - 1, j, mat[i][j]),
                                solve(mat, i, j + 1, mat[i][j]),
                                solve(mat, i, j - 1, mat[i][j]) });       
}

// Get the longest path in the DAG.(as wrap-around is not allowed DAG)
// Topological sort can iterate the vertices of a DAG in the linear ordering.
// Using Kahn''s algorithm(BFS) to implement topological sort while 
// counting the levels can give us the longest chain of nodes in the DAG.


int longestIncreasingPath(vector<vector>& matrix) {
int m = matrix.size() , n = matrix[0].size() ;
vector<vector> indegree(m , vector(n , 0)) ;
vector dir = {-1,0,1,0,-1} ;
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            for (int k = 0 ; k < 4 ; k++) {
                int next_i = i + dir[k] ; 
                int next_j = j + dir[k+1] ; 
                if (next_i >= 0 && next_j >= 0 && next_i < m && next_j < n 
                    && matrix[next_i][next_j] > matrix[i][j]) {
                    indegree[next_i][next_j]++ ; 
                }
            }
        }
    }
    
    queue<pair<int,int>> Q ;
    
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            if (indegree[i][j] == 0) {
                Q.push({i,j}) ; 
            }
        }
    }
    
    int maxlen = 0 ;
    
    while (!Q.empty()) {
        int size = Q.size() ; 
        for (int i = 0 ; i < size ; i++) {
            auto [ x , y ] = Q.front() ; Q.pop() ; 
            for (int k = 0 ; k < 4 ; k++) {
                int nx = x + dir[k] ; 
                int ny = y + dir[k+1] ; 
                if (nx >= 0 && ny >= 0 && nx < m && ny < n && matrix[nx][ny] > matrix[x][y]) {
                    indegree[nx][ny]-- ; 
                    if (indegree[nx][ny] == 0) {
                        Q.push({nx,ny}) ; 
                    }
                }
            }
        }
        maxlen++ ; 
    }
    
    return maxlen ; 
}

//python

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        rows = len(matrix)
        if rows == 0:
            return 0
        
        cols = len(matrix[0])
        indegree = [[0 for x in range(cols)] for y in range(rows)] 
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        
        for x in range(rows):
            for y in range(cols):
                for direction in directions:
                    nx, ny = x + direction[0], y + direction[1]
                    if nx >= 0 and ny >= 0 and nx < rows and ny < cols:
                        if matrix[nx][ny] < matrix[x][y]:
                            indegree[x][y] += 1
                            
        queue = []
        for x in range(rows):
            for y in range(cols):
                if indegree[x][y] == 0:
                    queue.append((x, y))
    
        path_len = 0
        while queue:
            sz = len(queue)
            for i in range(sz):
                x, y = queue.pop(0)
                for direction in directions:
                    nx, ny = x + direction[0], y + direction[1]
                    if nx >= 0 and ny >= 0 and nx < rows and ny < cols:
                        if matrix[nx][ny] > matrix[x][y]:
                            indegree[nx][ny] -= 1
                            if indegree[nx][ny] == 0:
                                queue.append((nx, ny))
            path_len += 1
        return path_len 


