72. Edit Distance
==================
// Given two strings word1 and word2, return the minimum number of operations 
// required to convert word1 to word2.

// You have the following three operations permitted on a word:

//     Insert a character
//     Delete a character
//     Replace a character

 

// Example 1:

// Input: word1 = "horse", word2 = "ros"
// Output: 3
// Explanation: 
// horse -> rorse (replace 'h' with 'r')
// rorse -> rose (remove 'r')
// rose -> ros (remove 'e')

// Example 2:

// Input: word1 = "intention", word2 = "execution"
// Output: 5
// Explanation: 
// intention -> inention (remove 't')
// inention -> enention (replace 'i' with 'e')
// enention -> exention (replace 'n' with 'x')
// exention -> exection (replace 'n' with 'c')
// exection -> execution (insert 'u')


// To apply DP, we define the state dp[i][j] to be the minimum number of 
// operations to convert word1[0..i) to word2[0..j).

// For the base case, that is, to convert a string to an empty string, the 
// mininum number of operations (deletions) is just the length of the string. 
// So we have dp[i][0] = i and dp[0][j] = j.

// For the general case to convert word1[0..i) to word2[0..j), we break this 
// problem down into sub-problems. Suppose we have already known how to convert 
// word1[0..i - 1) to word2[0..j - 1) (dp[i - 1][j - 1]), 
// if word1[i - 1] == word2[j - 1], then no more operation is needed and 
// dp[i][j] = dp[i - 1][j - 1].

// If word1[i - 1] != word2[j - 1], we need to consider three cases.

// Replace word1[i - 1] by word2[j - 1] (dp[i][j] = dp[i - 1][j - 1] + 1);
// If word1[0..i - 1) = word2[0..j) then delete word1[i - 1] 
// (dp[i][j] = dp[i - 1][j] + 1);
// If word1[0..i) + word2[j - 1] = word2[0..j) then insert word2[j - 1] to 
// word1[0..i) (dp[i][j] = dp[i][j - 1] + 1).

// So when word1[i - 1] != word2[j - 1], dp[i][j] will just be the minimum of the 
// above three cases.



int minDistance(string word1, string word2) {
        int n = word1.length(),m=word2.length();
        vector<vector<int>> dp(n+1,vector<int> (m+1,0));
        for(int i=0;i<=n;++i) dp[i][0]=i;
        for(int j=0;j<=m;++j) dp[0][j]=j;
        for(int i=1;i<=n;++i){
            for(int j=1;j<=m;++j){
                if(word1[i-1]==word2[j-1])
                    dp[i][j]=dp[i-1][j-1];
                else
                    dp[i][j]=1+min(dp[i-1][j],min(dp[i][j-1],dp[i-1][j-1]));
            }
        }
        return dp[n][m];
    }


// space optimized

int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size(), pre;
        vector<int> cur(n + 1, 0);
        for (int j = 1; j <= n; j++) {
            cur[j] = j;
        }
        for (int i = 1; i <= m; i++) {
            pre = cur[0];
            cur[0] = i;
            for (int j = 1; j <= n; j++) {
                int temp = cur[j];
                if (word1[i - 1] == word2[j - 1]) {
                    cur[j] = pre;
                } else {
                    cur[j] = min(pre, min(cur[j - 1], cur[j])) + 1;
                }
                pre = temp;
            }
        }
        return cur[n];
    }