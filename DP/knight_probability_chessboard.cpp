688. Knight Probability in Chessboard
=========================================
// On an n x n chessboard, a knight starts at the cell (row, column) and 
// attempts to make exactly k moves. The rows and columns are 0-indexed, so the 
// top-left cell is (0, 0), and the bottom-right cell is (n - 1, n - 1).

// A chess knight has eight possible moves it can make. 

// Each time the knight is to move, it chooses one of eight possible moves uniformly 
// at random (even if the piece would go off the chessboard) and moves there.

// The knight continues moving until it has made exactly k moves or has moved off the 
// chessboard.

// Return the probability that the knight remains on the board after it has stopped 
// moving.

 

// Example 1:

// Input: n = 3, k = 2, row = 0, column = 0
// Output: 0.06250
// Explanation: There are two moves (to (1,2), (2,1)) that will keep the knight 
// on the board.From each of those positions, there are also two moves that will 
// keep the knight on the board.
// The total probability the knight stays on the board is 0.0625.

// Example 2:

// Input: n = 1, k = 0, row = 0, column = 0
// Output: 1.00000


//recursive
private int[][]dir = new int[][]{{-2,-1},{-1,-2},{1,-2},{2,-1},{2,1},{1,2},{-1,2},{-2,1}};
public double knightProbability(int N, int K, int r, int c) {
    return find(N,K,r,c);
}
public double find(int N,int K,int r,int c){
    if(r < 0 || r > N - 1 || c < 0 || c > N - 1) return 0;
    if(K == 0)  return 1;
    double rate = 0;
    for(int i = 0;i < dir.length;i++){
        rate += 0.125 * find(N,K - 1,r + dir[i][0],c + dir[i][1]); //1/8=0.125
    }
    return rate;
}
//memoization
unordered_map<int, unordered_map<int, unordered_map<int, double>>>dp;
public:
    double knightProbability(int N, int K, int r, int c) {
        if(dp.count(r) && dp[r].count(c) && dp[r][c].count(K)) return dp[r][c][K];
        if(r < 0 || r >= N || c < 0 || c >= N) return 0;
        if(K == 0) return 1;
        double total = knightProbability(N, K - 1, r - 1, c - 2) + knightProbability(N, K - 1, r - 2, c - 1) 
                     + knightProbability(N, K - 1, r - 1, c + 2) + knightProbability(N, K - 1, r - 2, c + 1) 
                     + knightProbability(N, K - 1, r + 1, c + 2) + knightProbability(N, K - 1, r + 2, c + 1) 
                     + knightProbability(N, K - 1, r + 1, c - 2) + knightProbability(N, K - 1, r + 2, c - 1);
        double res = total / 8;
        dp[r][c][K] = res;
        return res;
    }

//dp
double knightProbability(int N, int K, int r, int c) {
        vector<vector<int>> dir{{1,2},{1,-2},{-1,2},{-1,-2},{2,1},{2,-1},{-2,1},{-2,-1}};
        vector<vector<double>> dp(N,vector<double>(N,0));
        dp[r][c]=1;
        for(int k=1;k<=K;k++){
            vector<vector<double>> tem(N,vector<double>(N,0));
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    for(int v=0;v<8;v++){
                        int x=i+dir[v][0];
                        int y=j+dir[v][1];
                        if(x<0 || y<0 || x>=N || y>=N){
                            continue;
                        }
                        tem[i][j]+=dp[x][y];
                    }
                }
            }
            dp=tem;
        }
        double rec=0;
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                rec+=dp[i][j];
            }
        double res=rec/pow(8,K);
        return res;
    }