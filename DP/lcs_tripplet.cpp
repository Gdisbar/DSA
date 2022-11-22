LCS (Longest Common Subsequence) of three strings
====================================================
// Given 3 strings of all having length < 100,the task is to find the 
// longest common sub-sequence in all three given sequences.

// Examples: 

// Input : str1 = "geeks"  
//         str2 = "geeksfor"  
//         str3 = "geeksforgeeks"
// Output : 5
// Longest common subsequence is "geeks"
// i.e., length = 5

// Input : str1 = "abcd1e2"  
//         str2 = "bc12ea"  
//         str3 = "bd1ea"
// Output : 3
// Longest common subsequence is "b1e" 
// i.e. length = 3.

// The idea is to take a 3D array to store the 
// length of common subsequence in all 3 given 
// sequences i. e., L[m + 1][n + 1][o + 1]

// 1- If any of the string is empty then there
//    is no common subsequence at all then
//            L[i][j][k] = 0

// 2- If the characters of all sequences match
//    (or X[i] == Y[j] ==Z[k]) then
//         L[i][j][k] = 1 + L[i-1][j-1][k-1]

// 3- If the characters of both sequences do 
//    not match (or X[i] != Y[j] || X[i] != Z[k] 
//    || Y[j] !=Z[k]) then
//         L[i][j][k] = max(L[i-1][j][k], 
//                          L[i][j-1][k], 
//                          L[i][j][k-1])



int lcsOf3( string X, string Y, string Z, int m,
                               int n, int o)
{
    int L[m+1][n+1][o+1];
 
    /* Following steps build L[m+1][n+1][o+1] in
       bottom up fashion. Note that L[i][j][k]
       contains length of LCS of X[0..i-1] and
       Y[0..j-1]  and Z[0.....k-1]*/
    for (int i=0; i<=m; i++)
    {
        for (int j=0; j<=n; j++)
        {
            for (int k=0; k<=o; k++)
            {
                if (i == 0 || j == 0||k==0)
                    L[i][j][k] = 0;
 
                else if (X[i-1] == Y[j-1] && X[i-1]==Z[k-1])
                    L[i][j][k] = L[i-1][j-1][k-1] + 1;
 
                else
                    L[i][j][k] = max(max(L[i-1][j][k],
                                         L[i][j-1][k]),
                                     L[i][j][k-1]);
            }
        }
    }

//recursion
 
string X = "AGGT12";
string Y = "12TXAYB";
string Z = "12XBA";
 
int dp[100][100][100];
 
/* Returns length of LCS for X[0..m-1], Y[0..n-1]
and Z[0..o-1] */
int lcsOf3(int i, int j,int k)
{
    if(i==-1||j==-1||k==-1)
        return 0;
    if(dp[i][j][k]!=-1)
        return dp[i][j][k];
     
    if(X[i]==Y[j] && Y[j]==Z[k])
        return dp[i][j][k] = 1+lcsOf3(i-1,j-1,k-1);
    else
        return dp[i][j][k] = max(max(lcsOf3(i-1,j,k),
                            lcsOf3(i,j-1,k)),lcsOf3(i,j,k-1));
}
 
// Driver code
int main()
{
    memset(dp, -1,sizeof(dp));
    int m = X.length();
    int n = Y.length();
    int o = Z.length();
 
    cout << "Length of LCS is " << lcsOf3(m-1,n-1,o-1);
// this code is contributed by Kushdeep Mittal
}