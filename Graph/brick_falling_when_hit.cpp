803. Bricks Falling When Hit
==================================
// You are given an m x n binary grid, where each 1 represents a brick and 0 
// represents an empty space. A brick is stable if:

//     It is directly connected to the top of the grid, or
//     At least one other brick in its four adjacent cells is stable.

// You are also given an array hits, which is a sequence of erasures we want to 
// apply. Each time we want to erase the brick at the location 
// hits[i] = (rowi, coli). The brick on that location (if it exists) will disappear. 
// Some other bricks may no longer be stable because of that erasure and will fall. 
// Once a brick falls, it is immediately erased from the grid (i.e., it does not land 
// on other stable bricks).

// Return an array result, where each result[i] is the number of bricks that will 
// fall after the ith erasure is applied.

// Note that an erasure may refer to a location with no brick, and if it does, no 
// bricks drop.

 

// Example 1:

// Input: grid = [[1,0,0,0],[1,1,1,0]], hits = [[1,0]]
// Output: [2]
// Explanation: Starting with the grid:
// [[1,0,0,0],
//  [1,1,1,0]]
// We erase the underlined brick at (1,0), resulting in the grid:
// [[1,0,0,0],
//  [0,1,1,0]]
// The two underlined bricks are no longer stable as they are no longer 
// connected to the top nor adjacent to another stable brick, so they will fall. 
// The resulting grid is:
// [[1,0,0,0],
//  [0,0,0,0]]
// Hence the result is [2].

// Example 2:

// Input: grid = [[1,0,0,0],[1,1,0,0]], hits = [[1,1],[1,0]]
// Output: [0,0]
// Explanation: Starting with the grid:
// [[1,0,0,0],
//  [1,1,0,0]]
// We erase the underlined brick at (1,1), resulting in the grid:
// [[1,0,0,0],
//  [1,0,0,0]]
// All remaining bricks are still stable, so no bricks fall. The grid remains 
// the same:
// [[1,0,0,0],
//  [1,0,0,0]]
// Next, we erase the underlined brick at (1,0), resulting in the grid:
// [[1,0,0,0],
//  [0,0,0,0]]
// Once again, all remaining bricks are still stable, so no bricks fall.
// Hence the result is [0,0].



// When won''t a brick drop?
// A brick will not drop if it connects to top or its adjacent bricks will not drop.
// That is, the brick will not drop if it belongs to the same connected component 
// with top.
// Problems related to connect can be solved by Disjoint Set

// We represent the top as 0, and any grid[x][y] as (x * cols + y + 1).
// We union 1-cells on the first row with 0, and any two adjacent 1-cells.

// There are n hits.
// Instead of checking all cells after each hit, we start from the last hit to the 
// first hit, restoring it and observing the change of bricksLeft - that is actually 
// the corresponding bricks dropped. change of bricksLeft = change of size(find(0))

class Solution {
    private static final int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    private int[][] grid;
    private int rows, cols;

    public int[] hitBricks(int[][] grid, int[][] hits) {
        rows = grid.length;
        cols = grid[0].length;
        this.grid = grid;
        
        DisjointSet ds = new DisjointSet(rows * cols + 1);
        
        /** Mark cells to hit as 2. */
        for (int[] hit : hits) {
            if (grid[hit[0]][hit[1]] == 1) grid[hit[0]][hit[1]] = 2;
        }
        
        /** Union around 1 cells. */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j ++) {
                if (grid[i][j] == 1) unionAround(i, j, ds);
            }
        }
        
        int numBricksLeft = ds.size[ds.find(0)]; // numBricksLeft after the last erasure.
        int i = hits.length - 1; // Index of erasure.
        int[] numBricksDropped = new int[hits.length]; // Number of bricks that will drop after each erasure.
        
        while (i >= 0) {
            int x = hits[i][0];
            int y = hits[i][1];
            if (grid[x][y] == 2) {
                grid[x][y] = 1; // Restore to last erasure.
                unionAround(x, y, ds);
                int newNumBricksLeft = ds.size[ds.find(0)];
                numBricksDropped[i] = Math.max(newNumBricksLeft - numBricksLeft - 1, 0); // Excluding the brick to erase.
                numBricksLeft = newNumBricksLeft;
            }
            i--;
        }
        
        return numBricksDropped;
    }
    
    private void unionAround(int x, int y, DisjointSet ds) {   
        int curMark = mark(x, y);
        
        for (int[] direction : directions) {
            int nx = x + direction[0];
            int ny = y + direction[1];
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && grid[nx][ny] == 1) {
                ds.union(curMark, mark(nx, ny));
            }
        }
        
        if(x == 0) ds.union(0, curMark); // Connect to the top of the grid.
    }
    
    private int mark(int x, int y) {
        return x * cols + y + 1;
    }
    
    class DisjointSet {
        int[] parent, size;
        
        public DisjointSet(int n) {
            parent = new int[n];
            size = new int[n];
            Arrays.fill(size, 1);
            for (int i = 0; i < n; i++) { // 0 indicates top of the grid.
                parent[i] = i;
            }
        }
        
        public int find(int x) {
            if (x == parent[x]) return x;
            return parent[x] = find(parent[x]);
        }
        
        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX != rootY) {
                parent[rootX] = rootY;
                size[rootY] += size[rootX];
            }
        }
    }
}



class Solution {
public:
    int dirs[5] = {1, 0, -1, 0, 1};
    vector<int> parents, rank;
    int m, n;
    vector<int> hitBricks(vector<vector<int>>& grid, vector<vector<int>>& hits) {
        m = grid.size(), n = grid[0].size();
        parents.resize(m * n + 1), iota(parents.begin(), parents.end(), 0);
        rank.resize(m * n + 1, 1);
        
        for (auto hit : hits) {
            if (grid[hit[0]][hit[1]] == 1)
                grid[hit[0]][hit[1]] = 2;
        }
        
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (grid[i][j] == 1)
                    unionAround(i, j, grid);
        
        int numBricksLeft = rank[find(0)];
        int i = hits.size() - 1;
        vector<int> ret(hits.size());
        
        while (i >= 0) {
            int x = hits[i][0], y = hits[i][1];
            if (grid[x][y] == 2) {
                grid[x][y] = 1;
                unionAround(x, y, grid);
                int newnumBricksLeft = rank[find(0)];
                ret[i] = max(0, newnumBricksLeft - numBricksLeft - 1);
                numBricksLeft = newnumBricksLeft;
            }
            i--;
        }
        return ret;
    }
    
    void unionAround(int x, int y, vector<vector<int>>& grid) {
        int mark = x * n + y + 1;
        for (int i = 0; i < 4; i++) {
            int new_x = x + dirs[i];
            int new_y = y + dirs[i + 1];
            if (new_x >= 0 && new_x < grid.size() && new_y >= 0 && new_y < grid[0].size() && grid[new_x][new_y] == 1) {
                union_(mark, new_x * n + new_y + 1);
            }
        }
        if (x == 0) union_(0, mark);
    }
    
    
    void union_(int i, int j) {
        int r1 = find(i);
        int r2 = find(j);
        if (r1 != r2) {
            parents[r1] = r2;
            rank[r2] += rank[r1];
        }
    }
    
    int find(int num) {
        if (parents[num] == num)
            return num;
        return parents[num] = find(parents[num]);
    }
};


class DSU:
    def __init__(self, size: int):
        self.size = [1]*size
        self.parents = [idx for idx in range(size)]        
    
    def find(self, cell) -> None:
        if self.parents[cell] != cell:
            self.parents[cell] = self.find(self.parents[cell])
        return self.parents[cell]
                
    def union(self, cell1, cell2) -> None:
        parent1, parent2 = self.find(cell1), self.find(cell2)
        
        if parent1 != parent2:
            self.parents[parent1] = parent2
            self.size[parent2] += self.size[parent1]

class Solution:
    def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
        self.grid = grid
        
        self.rows, self.cols = len(grid), len(grid[0])
        ds = DSU(self.rows*self.cols+1)
        
        # mark hits
        for row, col in hits:
            if grid[row][col] == 1:
                grid[row][col] = 2  
        
        # unionize bricks
        for row in range(self.rows):
            for col in range(self.cols):
                if grid[row][col] == 1:
                    self.union_around(ds, row, col)
        
        num_bricks_left = ds.size[ds.find(0)]
        num_bricks_dropped = [0]*len(hits)
        
        for idx in range(len(hits)-1,-1,-1):
            row, col = hits[idx]
            
            if grid[row][col] == 2:
                grid[row][col] = 1
                self.union_around(ds, row, col)
                new_num_bricks_left = ds.size[ds.find(0)]
                num_bricks_dropped[idx] = max(0, new_num_bricks_left-num_bricks_left-1)
                num_bricks_left = new_num_bricks_left
        
        return num_bricks_dropped
        
    def get_pos(self, row, col):
        return (row*self.cols) + col + 1
        
    def union_around(self, ds, row, col):
        curr_pos = self.get_pos(row, col)
        directions = [[-1,0],[1,0],[0,1],[0,-1]]

        for delta_row, delta_col in directions:
            new_row, new_col = row+delta_row, col+delta_col

            if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.grid[new_row][new_col] == 1:
                ds.union(curr_pos, self.get_pos(new_row, new_col))

        if row == 0: ds.union(0, curr_pos)