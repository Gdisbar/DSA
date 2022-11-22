Count All Palindromic Subsequence in a given String
=======================================================
// Find how many palindromic subsequences (need not necessarily be distinct) 
// can be formed in a given string. Note that the empty string is not considered a 
// palindrome. 

// Examples: 

// Input : str = "abcd"
// Output : 4
// Explanation :- palindromic  subsequence are : "a" ,"b", "c" ,"d" 

// Input : str = "aab"
// Output : 4
// Explanation :- palindromic subsequence are :"a", "a", "b", "aa"

// Input : str = "aaaa"
// Output : 15

//recursion
                    abca
                      |
                1 + abc + bca
                     |     |
                ab+bc-b   bc+ca-c
                /\ /\     /\ /\
                a bb c

int dp[n][n]={-1};
int helper(string s,int i=0,int j=n-1){
    if(i>j) return 0;
    if(i==j) return 1;
    if(dp[i][j]!=-1) return dp[i][j];
    if(s[i]==s[j]) 
        return dp[i][j]=1+helper(s,i+1,j)+helper(s,i,j-1);
    else // abc = ab+bc-b , bca = bc+ca-c
        return dp[i][j]=helper(s,i+1,j)+helper(s,i,j-1)-helper(s,i+1,j-1);
}

//dp
csp[0][0] = "a(0)"=1 //single char is palindrome
cps[0][1] = "a(0)","a(1)","aa(0-1)" = cps[0][0]+csp[1][1]+1=3 //palindrome
cps[0][2] = "a(0)", "a(1)", "b(2)", "aa(0-1)"=cps[0][0]+csp[1][1]+cps[2][2]+1
cps[1][1] = "a(1)"
cps[1][2] = "a(1)","b(2)"=cps[1][1]+cps[2][2]-cps[2][1]=1+1-0=2

1 3 4 
0 1 2 
0 0 1 


int countPS(string str){

    int N = str.length();
 
    // csp[i][j] = count of palindromes between s[i...j]
    int cps[N + 1][N + 1];
    memset(cps, 0, sizeof(cps));
 
    // palindromic subsequence of length 1
    for (int i = 0; i < N; i++)
        cps[i][i] = 1;
 
    // check subsequence of length L is palindrome or not
    for (int L = 2; L <= N; L++) {
        for (int i = 0; i <= N-L; i++) {
            int k = L + i - 1; 
            if (str[i] == str[k])
                cps[i][k]= cps[i][k - 1] + cps[i + 1][k] + 1;
            else
                cps[i][k] = cps[i][k - 1] + cps[i + 1][k]- cps[i + 1][k - 1];
        }
    }
 
    // return total palindromic subsequence
    return cps[0][N - 1];
}

730. Count Different Palindromic Subsequences
===============================================
// Given a string s, return the number of different non-empty palindromic 
// subsequences in s. Since the answer may be very large, return it modulo 10^9 + 7.

// A subsequence of a string is obtained by deleting zero or more characters 
// from the string.

// A sequence is palindromic if it is equal to the sequence reversed.

// Two sequences a1, a2, ... and b1, b2, ... are different if there is some i 
// for which ai != bi.


// Example 1:

// Input: s = "bccb"
// Output: 6
// Explanation: The 6 different non-empty palindromic subsequences are 
// 'b', 'c', 'bb', 'cc', 'bcb', 'bccb'.
// Note that 'bcb' is counted only once, even though it occurs twice.

// Example 2:

// Input: s = "abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba"
// Output: 104860361
// Explanation: There are 3104860382 different non-empty palindromic subsequences, 
// which is 104860361 modulo 10^9 + 7.

//dp[i][j] = no of palindrome in s[i:j]
//s = "bccb"

j,i 0 1 2 3  //bcc,ccb = 2+2-1=3 //dp[i][j]=dp[i][j-1]+dp[i+1][j]-dp[i+1][j-1];
0   1 2 3 6  //bc = 1+1 = 2, cb=1+1=2 , cc=1+1=2 ,dp[i][j]=dp[i][j-1]+dp[i+1][j];
1     1 2 3
2       1 2
3         1
// well the above formula i.e based on when begin & end char same does''nt 
// hold true for all palindrome subseq


if(low > high)
// consider the string from i to j is "a...a" "a...a"... where there is no 
//character 'a' inside the leftmost and rightmost 'a'
/* eg:  "aba" while i = 0 and j = 2:  dp[1][1] = 1 records the palindrome{"b"}, 
the reason why dp[i + 1][j  - 1] * 2 counted is that we count 
dp[i + 1][j - 1] one time as {"b"}, and additional time as {"aba"}. 
The reason why 2 counted is that we also  count {"a", "aa"}. 
So totally dp[i][j] record the palindrome: {"a", "b", "aa", "aba"}. 
 */ 
low==high
// consider the string from i to j is "a...a...a" where there is only one 
//character 'a' inside the leftmost and rightmost 'a'
/* eg:  "aaa" while i = 0 and j = 2: the dp[i + 1][j - 1] records the 
palindrome {"a"}.  the reason why dp[i + 1][j  - 1] * 2 counted is that 
we count dp[i + 1][j - 1] one time as {"a"}, and additional time as {"aaa"}. 
the reason why 1 counted is that  we also count {"aa"} that the first 'a' 
come from index i and the second come from index j. So totally dp[i][j] 
records {"a", "aa", "aaa"}
*/
low < high
// consider the string from i to j is "a...a...a... a" where there are at 
//least two character 'a' close to leftmost and rightmost 'a'
/* eg: "aacaa" while i = 0 and j = 4: the dp[i + 1][j - 1] records the 
palindrome {"a",  "c", "aa", "aca"}. the reason why dp[i + 1][j  - 1] * 2 
counted is that we count dp[i + 1][j - 1] one time as {"a",  "c", "aa", "aca"}, 
and additional time as {"aaa",  "aca", "aaaa", "aacaa"}.  Now there 
is duplicate :  {"aca"},which is removed by deduce dp[low + 1][high - 1]. 
So totally dp[i][j] record {"a",  "c", "aa", "aca", "aaa", "aaaa", "aacaa"}
*/
int countPalindromicSubsequences(string s) {
        int n = s.length(),mod=1000000007;
        vector<vector<int>> dp(n,vector<int>(n));
        for(int i = 0; i < n; i++){
            dp[i][i] = 1;   
        }
        for(int distance = 1; distance < n; distance++){
            for(int i = 0; i < n - distance; i++){
                int j = i + distance;
                if(s[i] == s[j]){
                    int lo = i + 1;
                    int hi = j - 1;
                    //avoid duplicate
                    while(lo <= hi && s[lo] != s[j]){ lo++;}
                    while(lo <= hi && s[hi] != s[j]){hi--;}
                    if(lo > hi){
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 2;  
                    } 
                    else if(lo == hi){
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 1;  
                    }
                    else{ //lo < hi
                        dp[i][j] = dp[i + 1][j - 1] * 2 - dp[lo + 1][hi - 1]; 
                    }
                }
                else{  //s[i]!=s[j]
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1];  
                }
                dp[i][j] = dp[i][j] < 0 ? dp[i][j] + mod : dp[i][j] % mod;
            }
        }
        return dp[0][n - 1];
    }


1930. Unique Length-3 Palindromic Subsequences
===============================================
// Given a string s, return the number of unique palindromes of length three 
// that are a subsequence of s.

// Note that even if there are multiple ways to obtain the same subsequence, 
// it is still only counted once.

// A palindrome is a string that reads the same forwards and backwards.

// A subsequence of a string is a new string generated from the original string 
// with some characters (can be none) deleted without changing the relative order 
// of the remaining characters.

//     For example, "ace" is a subsequence of "abcde".

 

// Example 1:

// Input: s = "aabca"
// Output: 3
// Explanation: The 3 palindromic subsequences of length 3 are:
// - "aba" (subsequence of "aabca")
// - "aaa" (subsequence of "aabca")
// - "aca" (subsequence of "aabca")

// Example 2:

// Input: s = "adc"
// Output: 0
// Explanation: There are no palindromic subsequences of length 3 in "adc".

// Example 3:

// Input: s = "bbcbaba"
// Output: 4
// Explanation: The 4 palindromic subsequences of length 3 are:
// - "bbb" (subsequence of "bbcbaba")
// - "bcb" (subsequence of "bbcbaba")
// - "bab" (subsequence of "bbcbaba")
// - "aba" (subsequence of "bbcbaba")


// We track the first and last occurence of each character.

// Then, for each character, we count unique characters between its first and 
// last occurence. That is the number of palindromes with that character in the 
// first and last positions.

// Example: abcbba, we have two unique chars between first and last a (c and b), 
// and two - between first and last b (b and c). No characters in between c so 
// it forms no palindromes.

// Complexity Analysis

//     Time: O(n). We go though the string fixed number of times.
//     Memory: O(1)
"bbcbaba"
 0123456
    a b c
f = 4 0 2
l = 6 5 2

i   f l res
0 b 4 6 0
1 b 0 5 1

int countPalindromicSubsequence(string s) {
    vector<int> first(26,INT_MAX),last(26);
    int res = 0;
    for (int i = 0; i < s.size(); ++i) {
        first[s[i] - 'a'] = min(first[s[i] - 'a'], i); //if we've got previously
        last[s[i] - 'a'] = i;
    }
    for (int i = 0; i < 26; ++i){
        if (first[i] < last[i]) { //
            cout<<i<<" "<<s[i]<<" "<<first[i]<<" "<<last[i]<<" "<<res<<endl;
            res += unordered_set<char>(begin(s) + first[i] + 1, begin(s) + last[i]).size();
        }
    }
    return res;
}
