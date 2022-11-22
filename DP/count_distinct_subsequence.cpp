115. Distinct Subsequences
=============================
// Given two strings s and t, return the number of distinct subsequences of s 
// which equals t.

// A string''s subsequence is a new string formed from the original string by 
// deleting some (can be none) of the characters without disturbing the remaining 
// characters'' relative positions. (i.e., ACE is a subsequence of "ABCDE" while 
// "AEC" is not).

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



// Well, a dynamic programming problem. Let''s first define its state dp[i][j] 
// to be the number of distinct subsequences of t[0..i - 1] in s[0..j - 1]. 
// Then we have the following state equations:

// General case 1: dp[i][j] = dp[i][j - 1] if t[i - 1] != s[j - 1];
// General case 2: dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] if t[i - 1] == s[j - 1];
// Boundary case 1: dp[0][j] = 1 for all j;
// Boundary case 2: dp[i][0] = 0 for all positive i.

// Now let''s give brief explanations to the four equations above.

// If t[i - 1] != s[j - 1], the distinct subsequences will not include s[j - 1] 
// and thus all the number of distinct subsequences will simply be those in 
// s[0..j - 2], which corresponds to dp[i][j - 1];
// If t[i - 1] == s[j - 1], the number of distinct subsequences include two 
// parts: those with s[j - 1] and those without;
// An empty string will have exactly one subsequence in any string :-)
// Non-empty string will have no subsequences in an empty string.


class Solution {
public:
    int numDistinct(string s, string t) {
        int m = t.length(), n = s.length();
        vector<vector<int>> dp(m + 1, vector<int> (n + 1, 0));
        for (int j = 0; j <= n; j++) dp[0][j] = 1;
        for (int j = 1; j <= n; j++)
            for (int i = 1; i <= m; i++)
                dp[i][j] = dp[i][j - 1] + (t[i - 1] == s[j - 1] ? dp[i - 1][j - 1] : 0);
        return dp[m][n];
    }
};  

// Notice that we keep the whole m*n matrix simply for dp[i - 1][j - 1]. 
// So we can simply store that value in a single variable and further 
// optimize the space complexity. The final code is as follows.

class Solution {
public:
    int numDistinct(string s, string t) {
        int m = t.length(), n = s.length();
        vector<int> cur(m + 1, 0);
        cur[0] = 1;
        for (int j = 1; j <= n; j++) { 
            int pre = 1;
            for (int i = 1; i <= m; i++) {
                int temp = cur[i];
                cur[i] = cur[i] + (t[i - 1] == s[j - 1] ? pre : 0);
                pre = temp;
            }
        }
        return cur[m];
    }
};

940. Distinct Subsequences II
=================================
// Given a string s, return the number of distinct non-empty subsequences of s. 
// Since the answer may be very large, return it modulo 10^9 + 7.
// A subsequence of a string is a new string that is formed from the original 
// string by deleting some (can be none) of the characters without disturbing the 
// relative positions of the remaining characters. (i.e., ace is a subsequence 
// of "abcde" while aec is not.

// Example 1:

// Input: s = "abc"
// Output: 7
// Explanation: The 7 distinct subsequences are "a", "b", "c", "ab", 
// "ac", "bc", and "abc".

// Example 2:

// Input: s = "aba"
// Output: 6
// Explanation: The 6 distinct subsequences are "a", "b", "ab", "aa", "ba", and "aba".

// Example 3:

// Input: s = "aaa"
// Output: 3
// Explanation: The 3 distinct subsequences are "a", "aa" and "aaa".

dp[i] represents the count of unique subsequence ends with S[i].
dp[i] is initialized to 1 for S[0 ... i]
For each dp[i], we define j from 0 to i - 1, we have:

    if s[j] != s[i], dp[i] += dp[j]
    if s[j] == s[i], do nothing to avoid duplicates.

Then result = sum(dp[0], ... dp[n - 1]) = sum(dp[i])
//Time complexity: O(n^2)

Furthermore, we can use a sum to represent sum(dp[0], ..., dp[i - 1]).
And also a count array, in which count[S[i] - 'a'] represents the 
count of presented subsequence ends with S[i].
Then dp[i] = sum - count[S[i] - 'a'].//count of distinct subseq end at i
//Time complexity: O(n)

for (int i = 0; i < n; i++) {
    int index = S[i] - 'a';
    dp[i] += sum - count[index]; //total unique subseq upto s[i]
    dp[i] = (dp[i] + M) % M;
    sum = (sum + dp[i]) % M;
    count[index] = (count[index] + dp[i]) % M; //update 
}

// space optimized : O(1)

for (int i = 0; i < n; i++) {
    int index = S.charAt(i) - 'a';
    int cur = (1 + sum - count[index] + M) % M;
    sum = (sum + cur) % M;
    count[index] = (count[index] + cur) % M; 
}

// s = "aba" --> sum==6
//sum-cnt[i] = sum upto s[i] excluding all previos occurrence of s[i]
// making the substriing distinct - simmilar to add a char at end
//initally dp[i]=1,when we do dp[i]+=sum-cnt[i],s[i] get included

//dp[i]+=sum-cnt[i] , sum+=dp[i] , cnt[idx]+=dp[i]

s[i]  cnt[s[i]]   sum dp[i] 
 a       0         0   1    // dp[0]=1+(0-0)=1,sum=0+1=1,cnt[a]=0+1=1
 b       0         1   1    // dp[1]=1+(1-0)=2,sum=1+2=3,cnt[b]=0+2=2
 a       1         3   1    // dp[2]=1+(3-1)=3,sum=3+3=6,cnt[a]=1+3=4

