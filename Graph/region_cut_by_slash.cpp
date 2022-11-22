959. Regions Cut By Slashes
=============================
// An n x n grid is composed of 1 x 1 squares where each 1 x 1 square consists 
// of a '/', '\', or blank space ' '. These characters divide the square into 
// contiguous regions.

// Given the grid grid represented as a string array, return the number of regions.

// Note that backslash characters are escaped, so a '\' is represented as '\\'.

 

// Example 1:

// Input: grid = [" /","/ "]
// Output: 2

// Example 2:

// Input: grid = [" /","  "]
// Output: 1

// Example 3:

// Input: grid = ["/\\","\\/"]
// Output: 5
// Explanation: Recall that because \ characters are escaped, "\\/" 
// refers to \/, and "/\\" refers to /\.



//TLE
// Intuition

// Split a cell in to 4 parts like this.
// We give it a number top is 0, right is 1, bottom is 2 left is 3.

// image
// (photo by @Sabbi_coder)

// Two adjacent parts in different cells are contiguous regions.
// In case '/', top and left are contiguous, botton and right are contiguous.
// In case '\\', top and right are contiguous, bottom and left are contiguous.
// In case ' ', all 4 parts are contiguous.

// Congratulation.
// Now you have another problem of counting the number of islands.

// Explanation

// DFS will be good enough to solve it.
// As I did in 947.Most Stones Removed with Same Row or Column
// I solved it with union find.

// Complexity

// Time O(N^2)
// Space O(N^2)

class Solution {
public:
    vector<int> f;
   int count,n;
   int find(int x){
       while(x!=f[x])
           f[x]=find(f[x]);
       return f[x];
   }
   void uni(int x,int y){
       x=find(x),y=find(y);
       if(x!=y){
           f[x]=y;
           count--;
       }
   }
    
   int g(int i,int j,int k){
       return (i*n+j)*4+k;
   }
    int regionsBySlashes(vector<string>& grid) {
        n = grid.size();
        count = n * n * 4;
        for (int i = 0; i < n * n * 4; ++i)
            f.push_back(i);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > 0) uni(g(i - 1, j, 2), g(i, j, 0));
                if (j > 0) uni(g(i , j - 1, 1), g(i , j, 3));
                if (grid[i][j] != '/') {
                    uni(g(i , j, 0), g(i , j,  1));
                    uni(g(i , j, 2), g(i , j,  3));
                }
                if (grid[i][j] != '\\') {
                    uni(g(i , j, 0), g(i , j,  3));
                    uni(g(i , j, 2), g(i , j,  1));
                }
            }
        }
        return count;
    }
};

def regionsBySlashes(self, grid):
        f = {}
        def find(x):
            f.setdefault(x, x)
            if x != f[x]:
                f[x] = find(f[x])
            return f[x]
        def union(x, y):
            f[find(x)] = find(y)

        for i in xrange(len(grid)):
            for j in xrange(len(grid)):
                if i:
                    union((i - 1, j, 2), (i, j, 0))
                if j:
                    union((i, j - 1, 1), (i, j, 3))
                if grid[i][j] != "/":
                    union((i, j, 0), (i, j, 1))
                    union((i, j, 2), (i, j, 3))
                if grid[i][j] != "\\":
                    union((i, j, 3), (i, j, 0))
                    union((i, j, 1), (i, j, 2))
        return len(set(map(find, f)))

// We can upscale the input grid to [n * 3][n * 3] grid and draw "lines" 
// there. Then, we can paint empty regions using DFS and count them. Picture 
// below says it all. Note that [n * 2][n * 2] grid does not work as "lines" 
// are too thick to identify empty areas correctly.

// This transform this problem into 200. Number of Islands, where lines ('1') 
// are the water, and rest ('0') is the land.



int dfs(vector<vector<int>> &g, int i, int j) {
    if (min(i, j) < 0 || max(i, j) >= g.size() || g[i][j] != 0)
        return 0;
    g[i][j] = 1;
    return 1 + dfs(g, i - 1, j) + dfs(g, i + 1, j) + dfs(g, i, j - 1) + dfs(g, i, j + 1);
}
int regionsBySlashes(vector<string>& grid) {
    int n = grid.size(), regions = 0;
    vector<vector<int>> g(n * 3, vector<int>(n * 3, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (grid[i][j] == '/') 
                g[i * 3][j * 3 + 2] = g[i * 3 + 1][j * 3 + 1] = g[i * 3 + 2][j * 3] = 1;
            else if (grid[i][j] == '\\') 
                g[i * 3][j * 3] = g[i * 3 + 1][j * 3 + 1] = g[i * 3 + 2][j * 3 + 2] = 1;
    for (int i = 0; i < n * 3; ++i)
        for (int j = 0; j < n * 3; ++j)
            regions += dfs(g, i, j) ? 1 : 0;    
    return regions;
}

//Python

class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        def dfs(i: int, j: int) -> int:
            if min(i, j) < 0 or max(i, j) >= len(g) or g[i][j] != 0:
                return 0
            g[i][j] = 1
            return 1 + dfs(i - 1, j) + dfs(i + 1, j) + dfs(i, j - 1) + dfs(i, j + 1)
        n, regions  = len(grid), 0
        g = [[0] * n * 3 for i in range(n * 3)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] == '/':
                    g[i * 3][j * 3 + 2] = g[i * 3 + 1][j * 3 + 1] = g[i * 3 + 2][j * 3] = 1
                elif grid[i][j] == '\\':
                    g[i * 3][j * 3] = g[i * 3 + 1][j * 3 + 1] = g[i * 3 + 2][j * 3 + 2] = 1
        for i in range(n * 3):
            for j in range(n * 3):
                regions += 1 if dfs(i, j) > 0 else 0
        return regions