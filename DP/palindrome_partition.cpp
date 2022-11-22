132. Palindrome Partitioning II
==================================
// Given a string s, partition s such that every substring of the partition 
// is a palindrome.

// Return the minimum cuts needed for a palindrome partitioning of s.

 

// Example 1:

// Input: s = "aab"
// Output: 1
// Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.

// Example 2:

// Input: s = "a"
// Output: 0

// Example 3:

// Input: s = "ab"
// Output: 1


// This can be solved by two points:

//     cut[i] is the minimum of cut[j - 1] + 1 (j <= i), if [j, i] is palindrome.
//     If [j, i] is palindrome, [j + 1, i - 1] is palindrome, and c[j] == c[i].

// The 2nd point reminds us of using dp (caching).

// a   b   a   |   c  c
//                 j  i
//        j-1  |  [j, i] is palindrome
//    cut(j-1) +  1



public int minCut(String s) {
    char[] c = s.toCharArray();
    int n = c.length;
    int[] cut = new int[n];
    boolean[][] pal = new boolean[n][n];
    
    for(int i = 0; i < n; i++) {
        int min = i;
        for(int j = 0; j <= i; j++) {
            if(c[j] == c[i] && (j + 1 > i - 1 || pal[j + 1][i - 1])) {
                pal[j][i] = true;  
                min = j == 0 ? 0 : Math.min(min, cut[j - 1] + 1);
            }
        }
        cut[i] = min;
    }
    return cut[n - 1];
}

// space optimized

int minCut(string s) {
        int n = s.size();
        vector<int> cut(n+1, 0);  // number of cuts for the first k characters
        for (int i = 0; i <= n; i++) cut[i] = i-1;
        for (int i = 0; i < n; i++) {
        	// odd length palindrome , [i-j,i+j]
            for (int j = 0; i-j >= 0 && i+j < n && s[i-j]==s[i+j] ; j++) 
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j]);
            // even length palindrome , [i-j+1,i+j+1]
            for (int j = 1; i-j+1 >= 0 && i+j < n && s[i-j+1] == s[i+j]; j++) 
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j+1]);
        }
        return cut[n];
    }