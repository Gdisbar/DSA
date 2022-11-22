Print all possible paths from top left to bottom right of a mXn matrix
=========================================================================
// The problem is to print all the possible paths from top left to bottom 
// right of a mXn matrix with the constraints that from each cell you can 
// either move only to right or down.

// Examples : 

// Input : 1 2 3
//         4 5 6
// Output : 1 4 5 6
//          1 2 5 6
//          1 2 3 6

// Input : 1 2 
//         3 4
// Output : 1 2 4
//          1 3 4



bool issafe(int r,int c,vector<vector<int>>& visited,int n,int m) {
  return (r < n and c <m and visited[r] !=-1 );  
}
 
 
void FindPaths(vector<vector<int>> &grid,int r,int c, int n,int m,
                            vector<int> &ans) {
  
  if(r == n-1 and c == m-1) {
    ans.push_back(grid[r]);
    display(ans);  // function to display the path stored in ans vector
    ans.pop_back(); // pop back because we need to backtrack to explore more path
    return ;
  } 
   
  // we will store the current value in ch and mark the visited place as -1
  int ch = grid[r];
  
  ans.push_back(ch); // push the path in ans array
  grid[r] = -1;  // mark the visited place with -1
   
  // if is it safe to take next downward step then take it
  if(issafe(r+1,c,grid,n,m)) {
    FindPaths(grid,r+1,c,n,m,ans);
  }
   
  // if is it safe to take next rightward step then take it
  if(issafe(r,c+1,grid,n,m)) {
    FindPaths(grid,r,c+1,n,m,ans);
  }
   
  // backtracking step we need to make values original so to we can visit 
  // it by some another path
  grid[r] = ch;
   
  // remove the current path element we explore
  ans.pop_back();
  return ;
}

//TC- O(2^n*m)   , SC – O(n) 

class Solution:
     
    def __init__(self):
        self.mapping = {}
     
    def printAllPaths(self, M, m, n):
        if not self.mapping.get((m,n)):
            if m == 1 and n == 1:
                return [M[m-1][n-1]]
            else:
                res = []
                if n > 1:
                    a = self.printAllPaths(M, m, n-1)
                    for i in a:
                        if not isinstance(i, list):
                            i = [i]
                        res.append(i+[M[m-1][n-1]])
                if m > 1:
                    b = self.printAllPaths(M, m-1, n)
                    for i in b:
                        if not isinstance(i, list):
                            i = [i]
                        res.append(i+[M[m-1][n-1]])
            self.mapping[(m,n)] = res
        return self.mapping.get((m,n))
 
M = [[1, 2, 3], [4, 5, 6], [7,8,9]]
m, n = len(M), len(M[0])
a = Solution()
res = a.printAllPaths(M, m, n)
for i in res:
    print(i)

// Output

// 1 4 7 8 9 
// 1 4 5 8 9 
// 1 4 5 6 9 
// 1 2 5 8 9 
// 1 2 5 6 9 
// 1 2 3 6 9 

// using bfs

// path upto that cell and cell's coordinates
struct info {
    vector<int> path;
    int i;
    int j;
};
 
void printAllPaths(vector<vector<int> >& maze){
    int n = maze.size();
    int m = maze[0].size();
    queue<info> q;
    q.push({ { maze[0][0] }, 0, 0 }); // pushing top-left cell into the queue
    while (!q.empty()) {
        info p = q.front();
        q.pop();
        // if we reached the bottom-right cell i.e destination 
        if (p.i == n - 1 && p.j == m - 1) {
            for (auto x : p.path)
                cout << x << " ";
            cout << "\n";
        }
        // if we are in the last row then only right movement is possible
        else if (p.i == n - 1) {
            vector<int> temp = p.path;
            // updating the current path
            temp.push_back(maze[p.i][p.j + 1]);
            q.push({ temp, p.i, p.j + 1 });
        }
 
        // if we are in the last column then only down movement is possible
        else if (p.j == m - 1) {
            vector<int> temp = p.path;
            // updating the current path
            temp.push_back(maze[p.i + 1][p.j]);
 
            q.push({ temp, p.i + 1, p.j });
        }
 
        // else both right and down movement are possible
        else { // right movement
            vector<int> temp = p.path;
            // updating the current path
            temp.push_back(maze[p.i][p.j + 1]);
 
            q.push({ temp, p.i, p.j + 1 });
 
            // down movement
            temp.pop_back();
            // updating the current path
            temp.push_back(maze[p.i + 1][p.j]);
 
            q.push({ temp, p.i + 1, p.j });
        }
    }
}
 


from collections import deque

// class info:
//     def __init__(self, path, i, j):
//         self.path = path
//         self.i = i
//         self.j = j
 
 
def printAllPaths(maze):
    n = len(maze)
    m = len(maze[0])
 
    q = deque()
    q.append(info([maze[0][0]], 0, 0))
 
    while len(q) > 0:
        p = q.popleft()
        if p.i == n - 1 and p.j == m - 1:
            for x in p.path:
                print(x, end=" ")
            print()
        elif p.i == n-1:
            temp = p.path[:]
            temp.append(maze[p.i][p.j+1])
            q.append(info(temp, p.i, p.j+1))
 
        elif p.j == m-1:
            temp = p.path[:]
            temp.append(maze[p.i+1][p.j])
            q.append(info(temp, p.i+1, p.j))

        else:
            temp = p.path[:]
            temp.append(maze[p.i][p.j + 1])
            q.append(info(temp, p.i, p.j + 1))

            temp = temp[:-1]

            temp.append(maze[p.i + 1][p.j])
            q.append(info(temp, p.i + 1, p.j))