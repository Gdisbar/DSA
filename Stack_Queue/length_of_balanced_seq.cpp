Length of Longest Balanced Subsequence
========================================
// Given a string S, find the length of the longest balanced subsequence in it. 
// A balanced string is defined as:- 

//     A null string is a balanced string.
//     If X and Y are balanced strings, then (X)Y and XY are balanced strings.

// Examples: 

// Input : S = "()())"
// Output : 4

// ()() is the longest balanced subsequence 
// of length 4.

// Input : s = "()(((((()"
// Output : 4

// LBS of substring str[i...j] : 
// If str[i] == str[j]
//     LBS(str, i, j) = LBS(str, i + 1, j - 1) + 2
// Else
//     LBS(str, i, j) = max(LBS(str, i, j), LBS(str, i, k) + LBS(str, k + 1, j))
//                      Where i <= k < j   


int maxLength(char s[], int n)
{
    int dp[n][n];
    memset(dp, 0, sizeof(dp));
 
    // Considering all balanced
    // substrings of length 2
    for (int i = 0; i < n - 1; i++)
        if (s[i] == '(' && s[i + 1] == ')')
            dp[i][i + 1] = 2;
 
    // Considering all other substrings
    for (int l = 2; l < n; l++) {
        for (int i = 0, j = l; j < n; i++, j++) {
            if (s[i] == '(' && s[j] == ')')
                dp[i][j] = 2 + dp[i + 1][j - 1];
 
            for (int k = i; k < j; k++)
                dp[i][j] = max(dp[i][j],dp[i][k] + dp[k + 1][j]);
        }
    }
 
    return dp[0][n - 1];
}

// Time Complexity : O(n^2) 
// Auxiliary Space : O(n^2)

int maxLength(char s[], int n)
{
    // As it's subsequence - assuming first
    // open brace would map to a first close
    // brace which occurs after the open brace
    // to make subsequence balanced and second
    // open brace would map to second close
    // brace and so on.
 
    // Variable to count all the open brace
    // that does not have the corresponding
    // closing brace.
    int invalidOpenBraces = 0;
 
    // To count all the close brace that
    // does not have the corresponding open brace.
    int invalidCloseBraces = 0;
 
    // Iterating over the String
    for (int i = 0; i < n; i++) {
        if (s[i] == '(') {
 
            // Number of open braces that
            // hasn't been closed yet.
            invalidOpenBraces++;
        }
        else {
            if (invalidOpenBraces == 0) {
 
                // Number of close braces that
                // cannot be mapped to any open
                // brace.
                invalidCloseBraces++;
            }
            else {
 
                // Mapping the ith close brace
                // to one of the open brace.
                invalidOpenBraces--;
            }
        }
    }
    return (
        n - (invalidOpenBraces+ invalidCloseBraces));
}
 