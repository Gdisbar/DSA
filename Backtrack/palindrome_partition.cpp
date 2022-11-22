131. Palindrome Partitioning
==============================
// Given a string s, partition s such that every substring of the partition is a 
// palindrome. Return all possible palindrome partitioning of s.

// A palindrome string is a string that reads the same backward as forward.

 

// Example 1:

// Input: s = "aab"
// Output: [["a","a","b"],["aa","b"]]

// Example 2:

// Input: s = "a"
// Output: [["a"]]


public class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        boolean[][] dp = new boolean[s.length()][s.length()];
        // TC : n^2
        for(int i = 0; i < s.length(); i++) {
            for(int j = 0; j <= i; j++) {
                if(s.charAt(i) == s.charAt(j) && (i - j <= 2 || dp[j+1][i-1])) {
                    dp[j][i] = true;
                }
            }
        }
        helper(res, new ArrayList<>(), dp, s, 0);
        return res;
    }
    // TC : 2^n
    private void helper(List<List<String>> res, List<String> path, 
    				boolean[][] dp, String s, int pos) {
        if(pos == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        
        for(int i = pos; i < s.length(); i++) {
            if(dp[pos][i]) {
                path.add(s.substring(pos,i+1));
                helper(res, path, dp, s, i+1);
                path.remove(path.size()-1);
            }
        }
    }
}


    def partition(self, s):
        out = []
        def isPalindrome(st):
            return st == st[::-1]
        
        def dfs(curr=[], index=1):
            if index == len(s) + 1:
                out.append(list(curr))
                return
            for i in range(index, len(s)+1):
                if isPalindrome(s[index-1:i]):
                    curr.append(s[index-1:i])
                    dfs(curr, i+1)
                    curr.pop()
        dfs()
        return out