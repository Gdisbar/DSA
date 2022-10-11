5. Longest Palindromic Substring
====================================
Given a string s, return the longest palindromic substring in s.

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:

Input: s = "cbbd"
Output: "bb"

//Brute force : find all substring (n*n) * find palindrome(n) = n^3

//DP , TC: n^2 , SC : n^2

s="aaaabbaa"

abcba is a palindrome if start--> abcba <--- end
    1. s[start]== s[end] 
    2. non-boundary substring is palindrome i.e dp[i+1][j-1] is palindrome

so s[i,j] is palindrome --> s[i]==s[j] && dp[i+1][j-1] is palindrome for len>=3

//when we find s[i]==s[j] we check dp[i+1][j-1]
 
i= 01234567
s="aaaabbaa"
s[0,2]="aaa" = s[i=0]==s[j=2] && s[i=0+1]==s[j=2-1] for s[1,3] same
s[2,4]="aab" = s[i=2]!=s[j=4]
s[3,5]="abb" = s[3]!=s[5]
s[4,6]="bba" = s[4]!=s[6]
s[5,7]="baa" = s[5]!=s[7]
-----------------------------
s[0,3]="aaaa" 
s[1,4]="aaab"
s[2,5]="aabb"
s[3,6]="abba" = s[i]==s[j] && dp[4][5]=1
s[4,7]="bbaa"
--------------------------------
s[0,4]="aaaab"
s[1,5]="aaabb"
s[2,6]="aabba" =s[i]==s[j] && dp[3][5]=0
s[3,7]="abbaa"
--------------------------------
s[0,5]="aaaabb"
s[1,6]="aaabba"
s[2,7]="aabbaa" = s[i]==s[j] && dp[3][6]=1 ----> longest palindrome
----------------------------------
s[0,6]="aaaabba" = s[i]==s[j] && dp[1][5]=0
s[1,7]="aaabbaa" = s[i]==s[j] && dp[2][6]=0
----------------------------------
s[0,7]="aaaabbaa" = s[i]==s[j] && dp[2][6]=0


//     Goal state
//     max(end - start + 1) for all state(start, end) = true

//     State transition

// for start = end (e.g. 'a'), state(start, end) is True
// for start + 1 = end (e.g. 'aa'), state(start, end) is True if s[start] = s[end]
// for start + 2 = end (e.g. 'aba'),  state(start, end) is True if s[start] = s[end] and state(start + 1, end - 1)
// for start + 3 = end (e.g. 'abba'),  state(start, end) is True if s[start] = s[end] and state(start + 1, end - 1)

//i & j loops are designed such a way that if s[i,j] is a match it checks if 
//in between chars form palindrome or not using dp[i+1][j-1] i.e travel backward


    string longestPalindrome(string s) {
        if(s.size()<=1) return s;
        int n = s.size();
        vector<vector<bool>> dp(n,vector<bool>(n,false));
        for(int i=0;i<n;++i) dp[i][i]=true;    
        int mxstart = 0;
        int mxlen = 1;
        for(int j=0;j<n;++j){
            for(int i=j-1;i>=0;--i){ //start from j=1 , i.e atleast string length >=1
                if(s[i]==s[j]){
                    if(j==i+1 or dp[i+1][j-1]){ // palindrom length = 2 or dp[i+1][j-1]==true
                        dp[i][j]=true;
                        int len=j-i+1;
                        if(mxlen<len){
                            mxstart=i;
                            mxlen=len;
                        }
                    }
                }
            }
        }

        return s.substr(mxstart,mxlen);
    }

  
(1) expand around every center(including inbetween '#') & check if palinderome is found
above one gives answer in n*n & it''s happening as we''re expanding around every centre

//expanding around every centers --> n*n
abababa --> [a]bababa,(1)--> [a|b|a]baba,(3)--> [ab|a|ba]ba,(5)--> [aba|b|aba],(7)

palindrom is symmetric only inside boundaries not outside of it, so adding new 
element in left or right doesn''t carried out as mirror image

after adding b to right , we have :
ababa[b|a|b](3) --->  aba[ba|b|ab](5)  --->   a[bab|a|bab](7)


1) len(mirror) goes beyond L , ?=R-B''s index
2) len(mirror) is within L , ?=len[mirror]
3) expand beyond minimum length from 1 & 2

// Manacher's algorithm , TC : 2*n , SC : n
    |L,mir|C    |R
T=$#a#b#a#b#a#b#a#@
P=00103050705030100

// https://www.youtube.com/watch?v=nbTSfrEfo6M ---> check out 1st / IDeserve

// 93% faster, 63% less memory

string longestPalindrome(string s) {
        vector<char> T(s.size()*2+3); //$ + 2*N+1 position + @ 
        T[0] = '$';
        T[s.size()*2 + 2] = '@';
        for (int i = 0; i < s.size(); i++) {
            T[2*i + 1] = '#';
            T[2*i + 2] = s[i];
        }
        T[s.size()*2 + 1] = '#';
         
         
        vector<int> P(T.size()); //2*N+3
        int center = 0, right = 0;
         
        for (int i = 1; i < T.size()-1; i++) { // i searching new mirror center 
            int mirr = 2*center - i;
 
            if (i < right) //checking if it''s inside right boundary,if yes case-1 + copy previous P[i]
                P[i] = min(right - i, P[mirr]);
          // match char about i
          while (T[i + (1 + P[i])] == T[i - (1 + P[i])])
                P[i]++;
            //change mirror center to current center & move right to end of palindrome
            if (i + P[i] > right) {
                center = i;
                right = i + P[i];
            }
        }
         
        int length = 0;    // length of longest palindromic substring
        center = 0;       // center of longest palindromic substring
        for (int i = 1; i < P.size()-1; i++) {
            if (P[i] > length) {
                length = P[i];
                center = i;
            }
        }
        return s.substr((center - 1 - length) / 2, length);
    }

