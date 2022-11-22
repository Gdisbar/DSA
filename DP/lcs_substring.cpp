1143. Longest Common Subsequence
====================================
// Given two strings text1 and text2, return the length of their longest 
// common subsequence. If there is no common subsequence, return 0.

// A subsequence of a string is a new string generated from the original 
// string with some characters (can be none) deleted without changing the 
// relative order of the remaining characters.

//     For example, "ace" is a subsequence of "abcde".

// A common subsequence of two strings is a subsequence that is common to both 
// strings.

 

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


// Recursive


 
// Returns length of function f
// or longest common substring
// of X[0..m-1] and Y[0..n-1]
int lcs(int i, int j, int count,string &X,string &Y)
{
 	// X = "abcdxyz";
  //   Y = "xyzabcd";
  //   n = X.size();
  //   m = Y.size();
  //   lcs(n, m, 0,X,Y)

    if (i == 0 || j == 0)
        return count;
 
    if (X[i - 1] == Y[j - 1]) {
        count = lcs(i - 1, j - 1, count + 1,X,Y);
    }
    count = max(count,max(lcs(i, j - 1, 0,X,Y),lcs(i - 1, j, 0,X,Y)));
    return count;
}

// SC : m*n
int longestCommonSubsequence(string s1, string s2) {
    int n=s1.size();
    int m=s2.size();

    vector<vector<int>> dp(n+1,vector<int>(m+1,-1));
    //idx1<1 
    for(int i=0;i<=n;i++){
        dp[i][0] = 0;
    }
    //idx2<1
    for(int i=0;i<=m;i++){
        dp[0][i] = 0;
    }
    
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(s1[i-1]==s2[j-1])
                dp[i][j] = 1 + dp[i-1][j-1];
            else
                dp[i][j] = 0 + max(dp[i-1][j],dp[i][j-1]);
        }
    }
    return dp[n][m];

    }

// SC : min(m,n)
int longestCommonSubsequence(string s1, string s2) {
        int n=s1.size(),m=s2.size();
        vector<int> prev(m+1,0),cur(m+1,0);
        for(int idx1=1;idx1<=n;++idx1){
            for(int idx2=1;idx2<=m;++idx2){
                if(s1[idx1-1]==s2[idx2-1])
                    cur[idx2]=1+prev[idx2-1];
                else cur[idx2]=0+max(prev[idx2],cur[idx2-1]);
            }
            prev=cur;
        }
        return prev[m];
    }

//better

int longestCommonSubsequence(string text1, string text2) {
        // LCS -> LIS
        vector<int> alph[128];  // record text1's alphabet in text2 pos.
        int maps[128];
        memset(maps, 0, sizeof(maps));
        for(int i = 0; i < text1.size(); i++) maps[text1[i]] = 1;
        
        for(int j = text2.size(); j > -1; j--) if(maps[text2[j]] == 1) alph[text2[j]].push_back(j);
        vector<int> nums;
        for(int i = 0; i < text1.size(); i++) {
            if(alph[text1[i]].size() > 0) nums.insert(nums.end(), alph[text1[i]].begin(), alph[text1[i]].end());
        }
        
        // get LIS's length by monotone stack method : O(nlogn)
        vector<int> pool;
        for(int i = 0; i < nums.size(); i++) {
            if(i == 0 || nums[i] > pool.back() ) {
                pool.push_back(nums[i]);
            } else if(nums[i] == pool.back()) {
                continue;
            } else {
                int s = 0, e = pool.size() - 1, mid = 0;
                while(s < e) {
                    mid = (s + e)/2;
                    if(pool[mid] < nums[i]) s = mid + 1;
                    else e = mid;
                }
                pool[e] = nums[i];
            }
        }
        
        return pool.size();
    }

# Python code for the above approach
from functools import lru_cache
from operator import itemgetter
 
def longest_common_substring(x: str, y: str) -> (int, int, int):
     
    # function to find the longest common substring
 
    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1) 
     
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:
       
        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0
 
    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():
         
        # upper right triangle of the 2D array
        for k in range(len(x)):       
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                    range(len(y) - 1, -1, -1)))
         
        # lower left triangle of the 2D array
        for k in range(len(y)):       
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                    range(len(x) - 1, -1, -1)))
 
    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))
 
# Driver Code
if __name__ == '__main__':
    x: str = 'GeeksforGeeks'
    y: str = 'GeeksQuiz'
    length, i, j = longest_common_substring(x, y)
    print(f'length: {length}, i: {i}, j: {j}')
    print(f'x substring: {x[i: i + length]}')
    print(f'y substring: {y[j: j + length]}')

Output

length: 5, i: 0, j: 0
x substring: Geeks
y substring: Geeks

Time Complexity: O(|X||Y|)
Space Complexity: O(1)
