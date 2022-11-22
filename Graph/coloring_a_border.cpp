1034. Coloring A Border
=========================
// You are given an m x n integer matrix grid, and three integers row, col, 
// and color. Each value in the grid represents the color of the grid square at 
// that location.

// Two squares belong to the same connected component if they have the same color 
// and are next to each other in any of the 4 directions.

// The border of a connected component is all the squares in the connected component 
// that are either 4-directionally adjacent to a square not in the component, or 
// on the boundary of the grid (the first or last row or column).

// You should color the border of the connected component that contains the square 
// grid[row][col] with color.

// Return the final grid.

 

// Example 1:

// Input: grid = [[1,1],[1,2]], row = 0, col = 0, color = 3
// Output: [[3,3],[3,2]]

// Example 2:

// Input: grid = [[1,2,2],[2,3,2]], row = 0, col = 1, color = 3
// Output: [[1,3,3],[2,3,3]]

// Example 3:

// Input: grid = [[1,1,1],[1,1,1],[1,1,1]], row = 1, col = 1, color = 2
// Output: [[2,2,2],[2,1,2],[2,2,2]]


class Solution {
vector<vector<int>> dir={{0,1},{1,0},{0,-1},{-1,0}};
private:
    void dfs(vector<vector<int>>& grid,int x,int y,int val){
        //if(c>0&&c<4)
        grid[x][y]=-val;
        int c=0;
        for(int i=0;i<4;++i){
            int dx=x+dir[i][0];
            int dy=y+dir[i][1];
            if(dx<0||dy<0||dx>=grid.size()||dy>=grid[0].size()||
              abs(grid[dx][dy])!=val) {continue;} //out of bound
            c++;
            if(grid[dx][dy]==val){
                dfs(grid,dx,dy,val);
            }  
        }
        if(c==4){
                grid[x][y]=val;
                //dfs(grid,dx-x,dy-y,val);
            }
    }
public:
    vector<vector<int>> colorBorder(vector<vector<int>>& grid, int row, int col, int color) {
        dfs(grid,row,col,grid[row][col]);
        for(int i=0;i<grid.size();++i){
            for(int j=0;j<grid[i].size();++j){
                if(grid[i][j]<0)
                    grid[i][j]=color;
            }
        }
        return grid;
    }
};

// Python BFS
def colorBorder(self, grid, r0, c0, color):
    m, n = len(grid), len(grid[0])
    bfs, component, border = [[r0, c0]], set([(r0, c0)]), set()
    for r0, c0 in bfs:
        for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
            r, c = r0 + i, c0 + j
            if 0 <= r < m and 0 <= c < n and grid[r][c] == grid[r0][c0]:
                if (r, c) not in component:
                    bfs.append([r, c])
                    component.add((r, c))
            else:
                border.add((r0, c0))
    for x, y in border: grid[x][y] = color
    return grid


//Python DFS

def colorBorder(self, grid, r0, c0, color):
    seen, m, n = set(), len(grid), len(grid[0])

    def dfs(x, y):
        if (x, y) in seen: return True
        if not (0 <= x < m and 0 <= y < n and grid[x][y] == grid[r0][c0]):
            return False
        seen.add((x, y))
        if dfs(x + 1, y) + dfs(x - 1, y) + dfs(x, y + 1) + dfs(x, y - 1) < 4:
            grid[x][y] = color
        return True
    dfs(r0, c0)
    return grid