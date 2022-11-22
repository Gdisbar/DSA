115. Distinct Subsequences
=============================
// Given two strings s and t, return the number of distinct subsequences of s 
// which equals t.

// A string's subsequence is a new string formed from the original string 
// by deleting some (can be none) of the characters without disturbing the 
// remaining characters' relative positions. (i.e., "ACE" is a subsequence of 
// "ABCDE" while "AEC" is not).

// The test cases are generated so that the answer fits on a 32-bit signed integer.

 

// Example 1:

// Input: s = "rabbbit", t = "rabbit"
// Output: 3
// Explanation:
// As shown below, there are 3 ways you can generate "rabbit" from s.
// rabbbit
// rabbbit
// rabbbit

// Example 2:

// Input: s = "babgbag", t = "bag"
// Output: 5
// Explanation:
// As shown below, there are 5 ways you can generate "bag" from s.
// babgbag
// babgbag
// babgbag
// babgbag
// babgbag

// dp[i][j] to be the number of distinct subsequences of t[0..i - 1] in s[0..j - 1]

// dp[i][j] = dp[i][j - 1] if t[i - 1] != s[j - 1]
// dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] if t[i - 1] == s[j - 1]
// Boundary case 1: dp[0][j] = 1 for all j;
// Boundary case 2: dp[i][0] = 0 for all positive i.


// If t[i - 1] != s[j - 1], the distinct subsequences will not include 
// s[j - 1] and thus all the number of distinct subsequences will simply be those 
// in s[0..j - 2], which corresponds to dp[i][j - 1];
// If t[i - 1] == s[j - 1], the number of distinct subsequences include two parts: 
// those with s[j - 1] and those without;
// An empty string will have exactly one subsequence in any string :-)
// Non-empty string will have no subsequences in an empty string.



int numDistinct(string s, string t) {
	int m = t.length(), n = s.length();
	vector<vector<int>> dp(m + 1, vector<int> (n + 1, 0));
	for (int j = 0; j <= n; j++) dp[0][j] = 1;
	for (int j = 1; j <= n; j++)
	    for (int i = 1; i <= m; i++)
	        dp[i][j] = dp[i][j - 1] + (t[i - 1] == s[j - 1] ? dp[i - 1][j - 1] : 0);
	return dp[m][n];
}