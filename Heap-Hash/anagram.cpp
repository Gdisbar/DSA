438. Find All Anagrams in a String
==========================================
//  Given two strings s and p, return an array of all the start indices of 
//  p''s anagrams in s. You may return the answer in any order.

// An Anagram is a word or phrase formed by rearranging the letters of a 
// different word or phrase, typically using all the original letters exactly 
// once.

 

// Example 1:

// Input: s = "cbaebabacd", p = "abc"
// Output: [0,6]
// Explanation:
// The substring with start index = 0 is "cba", which is an anagram of "abc".
// The substring with start index = 6 is "bac", which is an anagram of "abc".

// Example 2:

// Input: s = "abab", p = "ab"
// Output: [0,1,2]
// Explanation:
// The substring with start index = 0 is "ab", which is an anagram of "ab".
// The substring with start index = 1 is "ba", which is an anagram of "ab".
// The substring with start index = 2 is "ab", which is an anagram of "ab".


 vector<int> findAnagrams(string s, string p) {
        if (s.empty()) return {};
        vector<int> res, cnt(128, 0);
        int ns = s.size(), np = p.size(), i = 0;
        for (char c : p) ++cnt[c];
        while (i < ns) {
            bool success = true;
            vector<int> tmp = cnt;
            for (int j = i; j < i + np; ++j) {
                if (--tmp[s[j]] < 0) { 
                //check frequency of each char in current window [i,i+np-1]
                    success = false;
                    break;
                }
            }
            if (success) {
                res.push_back(i); 
            }
            ++i;
        }
        return res;
    }


49. Group Anagrams
======================
// Given an array of strings strs, group the anagrams together. 
// You can return the answer in any order.

// An Anagram is a word or phrase formed by rearranging the letters of 
// a different word or phrase, typically using all the original letters 
// exactly once.

 

// Example 1:

// Input: strs = ["eat","tea","tan","ate","nat","bat"]
// Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

// Example 2:

// Input: strs = [""]
// Output: [[""]]

// Example 3:

// Input: strs = ["a"]
// Output: [["a"]]

// m[t] 0 aet //eat
// m[t] 0 aet //tea
// m[t] 1 ant //tan
// m[t] 0 aet //ate
// m[t] 1 ant //nat
// m[t] 2 abt //bat



vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> res;
        unordered_map<string, int> m;
        for (string str : strs) {
            string t = str;
            sort(t.begin(), t.end());
            if (!m.count(t)) {
                m[t] = res.size();
                res.push_back({});
            }
            res[m[t]].push_back(str);
        }
        return res;
    }



813. Find Anagram Mappings 
================================
// Given two lists A and B, and B is an anagram of A. B is an anagram 
// of A means B is made by randomizing the order of the elements in A.
// We want to find an index mapping P, from A to B. A mapping P[i] = j 
// means the i-th element in A appears in B at index j.
// These lists A and B may contain duplicates. If there are multiple 
// answers, output any of them.


// Notice

//     A, B have equal lengths in range [1, 100].
//     A[i], B[i] are integers in range [0, 10^5].

// Example

// Example1

// Input:  A = [12, 28, 46, 32, 50] and B = [50, 12, 32, 46, 28]
// Output: [1, 4, 3, 2, 0]
// Explanation:
// As P[0] = 1 because the 0th element of A appears at B[1], and P[1] = 4 because 
// the 1st element of A appears at B[4], and so on.

// Example2

// Input:  A = [1, 2, 3, 4, 5] and B = [5, 4, 3, 2, 1]
// Output: [4, 3, 2, 1, 0]
// Explanation:
// As P[0] = 4 because the 0th element of A appears at B[4], and P[1] = 3 because 
// the 1st element of A appears at B[3], and so on.


vector<int> anagramMappings(vector<int>& A, vector<int>& B) {
        vector<int> res;
        unordered_map<int, int> m;
        for (int i = 0; i < B.size(); ++i) m[B[i]] = i;
        for (int num : A) res.push_back(m[num]);
        return res;
    }


Check if any anagram of a string is palindrome or not
========================================================
We have given an anagram string and we have to check whether it can be made 
palindrome or not. 

Examples: 

Input : geeksforgeeks 
Output : No
There is no palindrome anagram of 
given string

Input  : geeksgeeks
Output : Yes
There are palindrome anagrams of
given string. For example kgeesseegk

#define NO_OF_CHARS 256
 
/* function to check whether characters of a string
   can form a palindrome */
bool canFormPalindrome(string str)
{
    // Create a count array and initialize all
    // values as 0
    int count[NO_OF_CHARS] = { 0 };
 
    // For each character in input strings,
    // increment count in the corresponding
    // count array
    for (int i = 0; str[i]; i++)
        count[str[i]]++;
 
    // Count odd occurring characters
    int odd = 0;
    for (int i = 0; i < NO_OF_CHARS; i++) {
        if (count[i] & 1)
            odd++;
 
        if (odd > 1)
            return false;
    }
 
    // Return true if odd count is 0 or 1,
    return true;
}


Check if two strings are k-anagrams or not
==============================================
// Given two strings of lowercase alphabets and a value k, the task is to 
// ind if two strings are K-anagrams of each other or not.
// Two strings are called k-anagrams if following two conditions are true. 

//     Both have same number of characters.
//     Two strings can become anagram by changing at most k characters in a string.

// Examples :  

// Input:  str1 = "anagram" , str2 = "grammar" , k = 3
// Output:  Yes
// Explanation: We can update maximum 3 values and 
// it can be done in changing only 'r' to 'n' 
// and 'm' to 'a' in str2.

// Input:  str1 = "geeks", str2 = "eggkf", k = 1
// Output:  No
// Explanation: We can update or modify only 1 
// value but there is a need of modifying 2 characters. 
// i.e. g and f in str 2.

const int MAX_CHAR = 26;
 
// Function to check if str1 and str2 are k-anagram
// or not
bool areKAnagrams(string str1, string str2, int k)
{
    // If both strings are not of equal
    // length then return false
    int n = str1.length();
    if (str2.length() != n)
        return false;
 
    int hash_str1[MAX_CHAR] = {0};
 
    // Store the occurrence of all characters
    // in a hash_array
    for (int i = 0; i < n ; i++)
        hash_str1[str1[i]-'a']++;
 
    // Store the occurrence of all characters
    // in a hash_array
    int count = 0;
    for (int i = 0; i < n ; i++)
    {
        if (hash_str1[str2[i]-'a'] > 0)
            hash_str1[str2[i]-'a']--;
        else
            count++;
 
        if (count > k)
            return false;
    }
 
    // Return true if count is less than or
    // equal to k
    return true;
}