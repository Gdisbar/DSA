Find shortest safe route in a path with landmines
====================================================
// Given a path in the form of a rectangular matrix having few landmines 
// arbitrarily placed (marked as 0), calculate length of the shortest safe route 
// possible from any cell in the first column to any cell in the last column of the 
// matrix. We have to avoid landmines and their four adjacent cells 
// (left, right, above and below) as they are also unsafe. We are allowed to move to 
// only adjacent cells which are not landmines. i.e. the route cannot contains any 
// diagonal moves.

// Examples:  

// Input: 
// A 12 x 10 matrix with landmines marked as 0

// [ 1  1  1  1  1  1  1  1  1  1 ]
// [ 1  0  1  1  1  1  1  1  1  1 ]
// [ 1  1  1  0  1  1  1  1  1  1 ]
// [ 1  1  1  1  0  1  1  1  1  1 ]
// [ 1  1  1  1  1  1  1  1  1  1 ]
// [ 1  1  1  1  1  0  1  1  1  1 ]
// [ 1  0  1  1  1  1  1  1  0  1 ]
// [ 1  1  1  1  1  1  1  1  1  1 ]
// [ 1  1  1  1  1  1  1  1  1  1 ]
// [ 0  1  1  1  1  0  1  1  1  1 ]
// [ 1  1  1  1  1  1  1  1  1  1 ]
// [ 1  1  1  0  1  1  1  1  1  1 ]

// Output:  
// Length of shortest safe route is 13


#define R 12
#define C 10
 

int rowNum[] = { -1, 0, 0, 1 };
int colNum[] = { 0, -1, 1, 0 };
 
bool isSafe(int mat[R][C], int visited[R][C],int x, int y){
    if (mat[x][y] == 0 || visited[x][y])
        return false;
    return true;
}
 

bool isValid(int x, int y){
    if (x < R && y < C && x >= 0 && y >= 0)
        return true;
    return false;
}
 
// A function to mark all adjacent cells of landmines as unsafe. Landmines are 
// shown with number 0
void markUnsafeCells(int mat[R][C])
{
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            // if a landmines is found
            if (mat[i][j] == 0)
            {
              // mark all adjacent cells
              for (int k = 0; k < 4; k++)
                if (isValid(i + rowNum[k], j + colNum[k]))
                    mat[i + rowNum[k]][j + colNum[k]] = -1;
            }
        }
    }
 
    // mark all found adjacent cells as unsafe
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            if (mat[i][j] == -1)
                mat[i][j] = 0;
        }
    }
 
    // Uncomment below lines to print the path
    /*for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            cout << std::setw(3) << mat[i][j];
        }
        cout << endl;
    }*/
}
 


void findShortestPathUtil(int mat[R][C], int visited[R][C],
                          int i, int j, int &min_dist, int dist)
{
    // if destination is reached
    if (j == C-1)
    {
        // update shortest path found so far
        min_dist = min(dist, min_dist);
        return;
    }
 
    // if current path cost exceeds minimum so far
    if (dist > min_dist)
        return;
 
    visited[i][j] = 1;
 
    // Recurse for all safe adjacent neighbours
    for (int k = 0; k < 4; k++)
    {
        if (isValid(i + rowNum[k], j + colNum[k]) &&
            isSafe(mat, visited, i + rowNum[k], j + colNum[k]))
        {
            findShortestPathUtil(mat,visited, i+rowNum[k],j+colNum[k], min_dist, 
                                        dist + 1);
        }
    }
 
    // Backtrack
    visited[i][j] = 0;
}
 
// A wrapper function over findshortestPathUtil()
void findShortestPath(int mat[R][C])
{
    // stores minimum cost of shortest path so far
    int min_dist = INT_MAX;
    // dist --> stores current path cost
    int dist = 0;
    // create a boolean matrix to store info about
    // cells already visited in current route
    int visited[R][C];
 
    // mark adjacent cells of landmines as unsafe
    markUnsafeCells(mat);
 
    // start from first column and take minimum
    for (int i = 0; i < R; i++)
    {
        // if path is safe from current cell
        if (mat[i][0] == 1)
        {
            // initialize visited to false
            memset(visited, 0, sizeof visited);
 
            // find shortest route from (i, 0) to any
            // cell of last column (x, C - 1) where
            // 0 <= x < R
            findShortestPathUtil(mat, visited, i, 0,min_dist,dist);
 
            // if min distance is already found
            if(min_dist == C - 1)
                break;
        }
    }
 
    // if destination can be reached
    if (min_dist != INT_MAX)
        cout << "Length of shortest safe route is "<< min_dist;
 
    else // if the destination is not reachable
        cout << "Destination not reachable from given source";
}



import sys
 
R = 12
C = 10
 

rowNum = [ -1, 0, 0, 1 ]
colNum = [ 0, -1, 1, 0 ]
 
min_dist = sys.maxsize
 

def isSafe(mat, visited, x, y):
    if (mat[x][y] == 0 or visited[x][y]):
        return False
    return True
 

def isValid(x, y):
    if (x < R and y < C and x >= 0 and y >= 0):
        return True
    return False
 

def markUnsafeCells(mat):
    for i in range(R):
        for j in range(C):
            # If a landmines is found
            if (mat[i][j] == 0):
                # Mark all adjacent cells
                for k in range(4):
                    if (isValid(i + rowNum[k], j + colNum[k])):
                        mat[i + rowNum[k]][j + colNum[k]] = -1

    for i in range(R):
        for j in range(C):
            if (mat[i][j] == -1):
                mat[i][j] = 0
 
    """ Uncomment below lines to print the path
    /*
     * for (int i = 0; i < R; i++) {
     * for (int j = 0; j < C; j++) { cout <<
     * std::setw(3) << mat[i][j]; } cout << endl; }
     *"""
 

def findShortestPathUtil(mat, visited, i, j, dist):    
    global min_dist
    if (j == C - 1):       
        min_dist = min(dist, min_dist)
        return
    if (dist > min_dist):
        return
    visited[i][j] = True
    for k in range(4):
        if (isValid(i + rowNum[k], j + colNum[k]) and isSafe(mat, visited, i + rowNum[k], j + colNum[k])):
            findShortestPathUtil(mat, visited, i + rowNum[k], j + colNum[k], dist + 1)
    visited[i][j] = False
 
def findShortestPath(mat):
    global min_dist
    min_dist = sys.maxsize
    visited = [[False for i in range(C)] for j in range(R)]
    markUnsafeCells(mat)
    for i in range(R):
        if (mat[i][0] == 1):
            // # Find shortest route from (i, 0) to any
            // # cell of last column (x, C - 1) where
            // # 0 <= x < R
            findShortestPathUtil(mat, visited, i, 0, 0)
            // # If min distance is already found
            if (min_dist == C - 1):
                break

    if (min_dist != sys.maxsize):
        print("Length of shortest safe route is", min_dist)
    else:
        print("Destination not reachable from given source")

