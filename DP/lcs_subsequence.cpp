1143. Longest Common Subsequence
====================================
// Given two strings text1 and text2, return the length of their longest 
// common subsequence. If there is no common subsequence, return 0.

// A subsequence of a string is a new string generated from the original 
// string with some characters (can be none) deleted without changing the 
// relative order of the remaining characters.

//     For example, "ace" is a subsequence of "abcde".

// A common subsequence of two strings is a subsequence that is common 
// to both strings.

 

// Example 1:

// Input: text1 = "abcde", text2 = "ace" 
// Output: 3  
// Explanation: The longest common subsequence is "ace" and its length is 3.

// Example 2:

// Input: text1 = "abc", text2 = "abc"
// Output: 3
// Explanation: The longest common subsequence is "abc" and its length is 3.

// Example 3:

// Input: text1 = "abc", text2 = "def"
// Output: 0
// Explanation: There is no such common subsequence, so the result is 0.

// Intuition

// LCS is a well-known problem, and there are similar problems here:

//     1092. Shortest Common Supersequence
//     1062. Longest Repeating Substring
//     516. Longest Palindromic Subsequence

// Bottom-up DP utilizes a matrix m where we track LCS sizes for each combination 
// of i and j.

//     If a[i] == b[j], LCS for i and j would be 1 plus LCS till the i-1 and 
//     j-1 indexes.
//     Otherwise, we will take the largest LCS if we skip a charracter from 
//     one of the string (max(m[i - 1][j], m[i][j - 1]).

// This picture shows the populated matrix for "xabccde", "ace" test case.




int longestCommonSubsequence(string &a, string &b) {
    short m[1001][1001] = {};
    for (auto i = 0; i < a.size(); ++i)
        for (auto j = 0; j < b.size(); ++j)
            m[i + 1][j + 1] = a[i] == b[j] ? m[i][j] + 1 : max(m[i + 1][j], m[i][j + 1]);
    return m[a.size()][b.size()];
}

// Complexity Analysis

//     Time: O(nm), where n and m are the string sizes.
//     Memory: O(nm).

// Memory-Optimized Solution

// You may notice that we are only looking one row up in the solution above. 
// So, we just need to store two rows.



int longestCommonSubsequence(string &a, string &b) {
    short m[2][1000] = {};
    for (int i = 0; i < a.size(); ++i)
        for (int j = 0; j < b.size(); ++j)
            m[!(i % 2)][j + 1] = a[i] == b[j] ? m[i % 2][j] + 1 : max(m[i % 2][j + 1], m[!(i % 2)][j]);
    return m[a.size() % 2][b.size()];
}

// Complexity Analysis

//     Time: O(nm), where n and m are the string sizes.
//     Memory: O(min(n,m)), assuming that we will use a smaller string 
//     for the column dimension.


int longestCommonSubsequence(string text1, string text2) {
        int len1 = text1.length(), len2 = text2.length();
        int cur_val, pre_val;
        vector<int> dp(len2+1, 0);
        for(int i=1;i<=len1;++i) {
            pre_val = 0;
            for(int j=1;j<=len2;++j) {
                cur_val = dp[j];
                dp[j] = max({dp[j], dp[j-1], pre_val + (text1[i-1]==text2[j-1])});
                pre_val = cur_val;
            }
        }
        return dp[len2];
    }