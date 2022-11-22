2014. Longest Subsequence Repeated k Times
=============================================
// You are given a string s of length n, and an integer k. You are tasked to 
// find the longest subsequence repeated k times in string s.

// A subsequence seq is repeated k times in the string s if seq * k is a 
// subsequence of s, where seq * k represents a string constructed by 
// concatenating seq k times.


// Return the longest subsequence repeated k times in string s. If multiple such 
// subsequences are found, return the lexicographically largest one. If there is
// no such subsequence, return an empty string.

 

// Example 1:

// Input: s = "letsleetcode", k = 2
// Output: "let"
// Explanation: There are two longest subsequences repeated 2 times: 
//"let" and "ete"."let" is the lexicographically largest one.

// Example 2:

// Input: s = "bb", k = 2
// Output: "b"
// Explanation: The longest subsequence repeated 2 times is "b".

// Example 3:

// Input: s = "ab", k = 2
// Output: ""
// Explanation: There is no subsequence repeated 2 times. Empty string is 
// returned.

str = "letsleetcode" , k=2
s   =  "letleete"
cnt = {'c':1,'d':1,'e':4,'l':2,'o':1,'s':1,'t':2}
ans = "" t te te te te let let let let let
// inside dfs -> inserting & removing of each char starting from 'z'
After push : t ,count : 0 ,prefix : t  ---------->
After push : l ,count : 0 ,prefix : tl
-------------------------------------------
After pop : l ,count : 2 ,prefix : t   ---------->
After push : e ,count : 2 ,prefix : te ---------->
After push : l ,count : 0 ,prefix : tel
------------------------------------------
After pop : l ,count : 2 ,prefix : te   ---------->
After push : e ,count : 0 ,prefix : tee
------------------------------------------
After pop : e ,count : 2 ,prefix : te  ---------->
After pop : e ,count : 4 ,prefix : t   ---------->
After pop : t ,count : 2 ,prefix :     ---------->
------------------------------------------
After push : l ,count : 0 ,prefix : l
After push : t ,count : 0 ,prefix : lt
After push : e ,count : 2 ,prefix : lte
-------------------------------------------
After pop : e ,count : 4 ,prefix : lt
After pop : t ,count : 2 ,prefix : l
-------------------------------------------
After push : e ,count : 2 ,prefix : le
After push : t ,count : 0 ,prefix : let  ---------->
After push : e ,count : 0 ,prefix : lete
------------------------------------------
After pop : e ,count : 2 ,prefix : let  ---------->
After pop : t ,count : 2 ,prefix : le
After push : e ,count : 0 ,prefix : lee
-------------------------------------------
After pop : e ,count : 2 ,prefix : le
After pop : e ,count : 4 ,prefix : l
After pop : l ,count : 2 ,prefix : 
-------------------------------------------
After push : e ,count : 2 ,prefix : e
After push : t ,count : 0 ,prefix : et
After push : l ,count : 0 ,prefix : etl
-------------------------------------------
After pop : l ,count : 2 ,prefix : et
After push : e ,count : 0 ,prefix : ete ---------->
After push : l ,count : 0 ,prefix : etel
-------------------------------------------
After pop : l ,count : 2 ,prefix : ete ---------->
After pop : e ,count : 2 ,prefix : et
After pop : t ,count : 2 ,prefix : e
-------------------------------------------
After push : l ,count : 0 ,prefix : el
-------------------------------------------
After pop : l ,count : 2 ,prefix : e
After push : e ,count : 0 ,prefix : ee
After push : t ,count : 0 ,prefix : eet
-------------------------------------------
After pop : t ,count : 2 ,prefix : ee
After push : l ,count : 0 ,prefix : eel
-------------------------------------------
After pop : l ,count : 2 ,prefix : ee
After pop : e ,count : 2 ,prefix : e
After pop : e ,count : 4 ,prefix :     ---------->


class Solution {
public:
    int n,k;
    string s,ans,prefix;
    vector<int> cnt;
    
    bool valid()
    {
        if (prefix.empty()) return true;
        if (prefix.size() > (n/k)) return false;
        int k1 = 0;
        int j = 0;
        for (int i = 0; i < n; i++)
        {
            if (prefix[j] == s[i])
            {
                j++;
                if (j == prefix.size())
                {
                    k1++;
                    j = 0;
                }
            }
        }
        return k1 >= k;
    }
    
    void dfs()
    {
        if (valid() == false) return;
        if (ans.size() < prefix.size()) ans = prefix;
        for (char c = 'z'; c >= 'a';c--)
        {
            int& d = cnt[c -'a'];
            if (d < k) continue;
            d -= k;
            prefix.push_back(c);
            dfs();
            prefix.pop_back();
            d += k;
        }
    }
    
    string longestSubsequenceRepeatedK(string str, int k1) {
        n = str.size();
        k = k1;
        cnt = vector<int>(27, 0);
        for (auto c : str) {
            cnt[c - 'a']++;
        }
        for (auto c : str) {
            if (cnt[c - 'a'] < k) continue;
            s.push_back(c);
        }
        n = s.size();
        prefix.reserve(8); //
        dfs();
        return ans;
    }
};

//slower

//Algorithm
// Count all characters in the input string.
// Collect characters that appear k times into chars.
//     More specifically, add character i to chars this number of times: cnt[i] / k.
//     Do it in a reverse direction (from z to a) so we process subsequences in a r
//     everse lexicographic order.
// Recursively generate and check subsequences.
// Pick a character (from largest to smallest), and add it to the current subsequence 
// (new_cur = cur + chars[i]).
// check to see if new_cur is repeated k times.
//     If so, continue generation recursivelly for new_cur.
//     Use a mask to mark used characters.
// class Solution {
// public:
//         bool check(string &s, string &p, int k) {
//             for (int i = 0, j = 0; i < s.size() && k > 0; ++i) {
//                 j += p[j] == s[i];
//                 if (j == p.size()) {
//                     --k;
//                     j = 0;
//                 }
//             }
//             return k == 0;
//         }
//         void generate(string &s, string &chars, string &cur, string &best, int mask, int k) {
//             for (int i = 0; i < chars.size(); ++i) {
//                 if ((mask & (1 << i)) == 0) {
//                     string new_cur = cur + chars[i];
//                     if (check(s, new_cur, k)) {
//                         if (new_cur.size() > best.size())
//                             best = new_cur;
//                         generate(s, chars, new_cur, best, mask + (1 << i), k);
//                     }
//                 }
//             }
//         }
//         string longestSubsequenceRepeatedK(string &s, int k) {
//             int cnt[26] = {};
//             string chars, best;
//             for (auto ch : s)
//                 ++cnt[ch - 'a'];
//             for (int i = 25; i >= 0; --i)
//                 chars += string(cnt[i] / k, 'a' + i);
//             string tmp="";
//             generate(s, chars, tmp, best, 0, k);
//             return best;
//         }

// };