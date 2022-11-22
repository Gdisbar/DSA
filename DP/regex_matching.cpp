10. Regular Expression Matching
================================
// Given an input string s and a pattern p, implement regular expression 
// matching with support for '.' and '*' where:

//     '.' Matches any single character.
//     '*' Matches zero or more of the preceding element.

// The matching should cover the entire input string (not partial).

 

// Example 1:

// Input: s = "aa", p = "a"
// Output: false
// Explanation: "a" does not match the entire string "aa".

// Example 2:

// Input: s = "aa", p = "a*"
// Output: true
// Explanation: '*' means zero or more of the preceding element, 'a'. 
// Therefore, by repeating 'a' once, it becomes "aa".

// Example 3:

// Input: s = "ab", p = ".*"
// Output: true
// Explanation: ".*" means "zero or more (*) of any character (.)".

// dp[i][j] denotes if s.substring(0,i) is valid for pattern p.substring(0,j)

// What about the first row? In other words which pattern p matches empty 
// string s=""? The answer is either an empty pattern p="" or a pattern that can 
// represent an empty string such as p="a*", p="z*" or more interestingly a 
// combiation of them as in p="a*b*c*". Below for loop is used to populate dp[0][j]. 
// Note how it uses previous states by checking dp[0][j-2]

// (p.charAt(j-1) == s.charAt(i-1) || p.charAt(j-1) == '.') if the current characters 
// match or pattern has . then the result is determined by the previous state 
// dp[i][j] = dp[i-1][j-1]. Don't be confused by the charAt(j-1) charAt(i-1) 
// indexes using a -1 offset that is because our dp array is actually one index 
// bigger than our string and pattern lenghts to hold the initial state dp[0][0]

// if p.charAt(j-1) == '*' then either it acts as an empty set and the result is 
// dp[i][j] = dp[i][j-2] or (s.charAt(i-1) == p.charAt(j-2) || p.charAt(j-2) == '.') 
// current char of string equals the char preceding * in pattern so the result is 
// dp[i-1][j]


bool isMatch(string s, string p) {
        if (p.length() == 0) return s.length() == 0;
        vector<vector<bool>> dp(s.length()+1,vector<bool>(p.length()+1));
        // as we need to hold dp[0][0] so length of dp is 1 more than length od s & p
        dp[0][0] = true;
        //first row
        for (int j=2; j<=p.length(); j++) {
        // uses previous state for checking p="a*b*c*
            dp[0][j] = p[j-1] == '*' && dp[0][j-2]; 
        }
        
        for (int j=1; j<=p.length(); j++) {
            for (int i=1; i<=s.length(); i++) {
                if (p[j-1] == s[i-1] || p[j-1] == '.') 
					dp[i][j] = dp[i-1][j-1];
                else if(p[j-1] == '*') //empty set or [i-1,j-1] result
                    dp[i][j] = dp[i][j-2] || ((s[i-1] == p[j-2] || p[j-2] == '.') && dp[i-1][j]); 
            }
        }
        return dp[s.length()][p.length()];
    }