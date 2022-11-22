The Knight’s tour problem
==============================
// Given a N*N board with the Knight placed on the first block of an empty board. 
// Moving according to the rules of chess knight must visit each square exactly 
// once. Print the order of each cell in which they are visited.

// Example:

// Input : 
// N = 8
// Output:
// 0  59  38  33  30  17   8  63
// 37  34  31  60   9  62  29  16
// 58   1  36  39  32  27  18   7
// 35  48  41  26  61  10  15  28
// 42  57   2  49  40  23   6  19
// 47  50  45  54  25  20  11  14
// 56  43  52   3  22  13  24   5
// 51  46  55  44  53   4  21  12


#define N 8
 
int isSafe(int x, int y, int sol[N][N]){
    return (x >= 0 && x < N && y >= 0 && y < N && sol[x][y] == -1);
}

int solveKTUtil(int x,int y,int pos,int sol[N][N],int xMove[8],int yMove[8]){
    int k, next_x, next_y;
    if (pos == N * N)
        return 1;
    for (k = 0; k < 8; k++) {
        next_x = x + xMove[k];
        next_y = y + yMove[k];
        if (isSafe(next_x, next_y, sol)) {
            sol[next_x][next_y] = pos;
            if (solveKTUtil(next_x, next_y, pos + 1, sol,xMove, yMove)== 1)
                return 1;
            else   // backtracking
                sol[next_x][next_y] = -1;
        }
    }
    return 0;
}

int solveKT(){
    int sol[N][N];
 
    /* Initialization of solution matrix */
    for (int x = 0; x < N; x++)
        for (int y = 0; y < N; y++)
            sol[x][y] = -1;
    int xMove[8] = { 2, 1, -1, -2, -2, -1, 1, 2 };
    int yMove[8] = { 1, 2, 2, 1, -1, -2, -2, -1 };
 
    // Since the Knight is initially at the first block
    sol[0][0] = 0;
    if (solveKTUtil(0, 0, 1, sol, xMove, yMove) == 0) {
        cout << "Solution does not exist";
        return 0;
    }
    else
        printSolution(sol);
 
    return 1;
}
 

 
 // TC :8^(N^2) -> N^2 cells with 8 possible moves to choose from


n = 8

def isSafe(x, y, board):
    if(x >= 0 and y >= 0 and x < n and y < n and board[x][y] == -1):
        return True
    return False
 
def solveKT(n):
    board = [[-1 for i in range(n)]for i in range(n)]
    move_x = [2, 1, -1, -2, -2, -1, 1, 2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]
    board[0][0] = 0
    pos = 1
    if(not solveKTUtil(n, board, 0, 0, move_x, move_y, pos)):
        print("Solution does not exist")
    else:
        printSolution(n, board)
 
def solveKTUtil(n, board, curr_x, curr_y, move_x, move_y, pos):
    if(pos == n**2):
        return True
    for i in range(8):
        new_x = curr_x + move_x[i]
        new_y = curr_y + move_y[i]
        if(isSafe(new_x, new_y, board)):
            board[new_x][new_y] = pos
            if(solveKTUtil(n, board, new_x, new_y, move_x, move_y, pos+1)):
                return True
            board[new_x][new_y] = -1
    return False
 

if __name__ == "__main__":
     
    # Function Call
    solveKT(n)