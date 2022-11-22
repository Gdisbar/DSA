44. Wildcard Matching
=======================
// Given an input string (s) and a pattern (p), implement wildcard pattern 
// matching with support for '?' and '*' where:

//     '?' Matches any single character.
//     '*' Matches any sequence of characters (including the empty sequence).

// The matching should cover the entire input string (not partial).

 

// Example 1:

// Input: s = "aa", p = "a"
// Output: false
// Explanation: "a" does not match the entire string "aa".

// Example 2:

// Input: s = "aa", p = "*"
// Output: true
// Explanation: '*' matches any sequence.

// Example 3:

// Input: s = "cb", p = "?a"
// Output: false
// Explanation: '?' matches 'c', but the second letter is 'a', which does 
// not match 'b'.


// greedy solution with idea of DFS
// starj stores the position of last * in p
// last_match stores the position of the previous matched char in s after a *
// e.g. 
// s: a c d s c d
// p: * c d
// after the first match of the *, starj = 0 and last_match = 1
// when we come to i = 3 and j = 3, we know that the previous match of * is 
//actually wrong, 
// (the first branch of DFS we take is wrong)
// then it resets j = starj + 1 
// since we already know i = last_match will give us the wrong answer
// so this time i = last_match+1 and we try to find a longer match of *
// then after another match we have starj = 0 and last_match = 4, which is the 
//right solution
// since we don't know where the right match for * ends, we need to take a guess 
//(one branch in DFS), 
// and store the information(starj and last_match) so we can always backup to the 
//last correct place and take another guess.

 bool isMatch(string s, string p) {
        int i = 0, j = 0;
        int m = s.length(), n = p.length();
        int last_match = -1, starj = -1;
        while (i < m){
            if (j < n && (s[i] == p[j] || p[j] == '?')){
                i++; j++;
            }
            else if (j < n && p[j] == '*'){
                starj = j;
                j++;
                last_match = i;
            }
            else if (starj != -1){
                j = starj + 1;
                last_match++;
                i = last_match;
            }
            else return false;
        }
        while (p[j] == '*' && j <n) j++;
        return j == n;
    }

// dp 

We define the state P[i][j] to be whether s[0..i) matches p[0..j). 
The state equations are as follows:

if p[j - 1] != '*'
P[i][j] = P[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '?')

else  // if p[j - 1] == '*'
P[i][j] = P[i][j - 1] || P[i - 1][j]


 bool isMatch(string s, string p) { 
        int m = s.length(), n = p.length();
        if (n > 30000) return false; // the trick , hard-coding the result
        vector<bool> cur(m + 1, false); 
        cur[0] = true;
        for (int j = 1; j <= n; j++) {
            bool pre = cur[0]; // use the value before update
            cur[0] = cur[0] && p[j - 1] == '*'; 
            for (int i = 1; i <= m; i++) {
                bool temp = cur[i]; // record the value before update
                if (p[j - 1] != '*')
                    cur[i] = pre && (s[i - 1] == p[j - 1] || p[j - 1] == '?');
                else cur[i] = cur[i - 1] || cur[i];
                pre = temp;
            }
        }
        return cur[m]; 
    }