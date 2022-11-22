753. Cracking the Safe
===========================
// There is a safe protected by a password. The password is a sequence of n digits 
// where each digit can be in the range [0, k - 1].

// The safe has a peculiar way of checking the password. When you enter in a 
// sequence, it checks the most recent n digits that were entered each time you 
// type a digit.

// For example, the correct password is "345" and you enter in "012345":
// After typing 0, the most recent 3 digits is "0", which is incorrect.
// After typing 1, the most recent 3 digits is "01", which is incorrect.
// After typing 2, the most recent 3 digits is "012", which is incorrect.
// After typing 3, the most recent 3 digits is "123", which is incorrect.
// After typing 4, the most recent 3 digits is "234", which is incorrect.
// After typing 5, the most recent 3 digits is "345", which is correct and the 
// safe unlocks.

// Return any string of minimum length that will unlock the safe at some 
// point of entering it.

 

// Example 1:

// Input: n = 1, k = 2
// Output: "10"
// Explanation: The password is a single digit, so enter each digit. "01" 
// would also unlock the safe.

// Example 2:

// Input: n = 2, k = 2
// Output: "01100"
// Explanation: For each possible password:
// - "00" is typed in starting from the 4th digit.
// - "01" is typed in starting from the 1st digit.
// - "10" is typed in starting from the 3rd digit.
// - "11" is typed in starting from the 2nd digit.
// Thus "01100" will unlock the safe. "01100", "10011", and "11001" would also 
// unlock the safe.


// In order to guarantee to open the box at last, the input password ought to 
// contain all length-n combinations on digits [0..k-1] - there should be k^n 
// combinations in total.

// To make the input password as short as possible, we''d better make each 
// possible length-n combination on digits [0..k-1] occurs exactly once as a 
// substring of the password. The existence of such a password is proved by 
// De Bruijn sequence:

// A de Bruijn sequence of order n on a size-k alphabet A is a cyclic sequence 
// in which every possible length-n string on A occurs exactly once as a 
// substring. It has length k^n, which is also the number of distinct substrings 
// of length n on a size-k alphabet; de Bruijn sequences are therefore optimally 
// short.

// We reuse last n-1 digits of the input-so-far password as below:

// e.g., n = 2, k = 2
// all 2-length combinations on [0, 1]: 
// 00 (`00`110), 
//  01 (0`01`10), 
//   11 (00`11`0), 
//    10 (001`10`)
   
// the password is 00110

// We can utilize DFS to find the password:

// goal: to find the shortest input password such that each possible n-length 
// combination of digits [0..k-1] occurs exactly once as a substring.

// node: current input password

// edge: if the last n - 1 digits of node1 can be transformed to node2 by 
// appending a digit from 0..k-1, there will be an edge between node1 and node2

// start node: n repeated 0's
// end node: all n-length combinations among digits 0..k-1 are visited

// visited : all combinations that have been visited

// -> why a path is guaranteed to be found?

// Let''s say there are 'p' permutations present and all are mutually 
// exclusive then they can just be concatenated and solution is found.

// Now let''s say the strings are overlapping, but what do we mean by overlapping?

// for any two strings S1 and S2,

// last 'd' characters of S1 = first 'd' characters of S2

// which means if we remove first 'd' characters of S2 and append remaining 
// "suffix" to S1 then we will have both strings present at once.

// Now extend this logic by chaining multiple such strings.

// Sidenote -
// All the permutations in this question might not chain in which case 
// we can put those augmented strings in any order, which is one of the 
// reasons multiple solutions are possible.

class Solution {
public:
    int n, k, total;
    
    bool backtrack(string& ans, unordered_set<string>& visited){
        if(visited.size() == total){
            return true;
        }else{
            for(int i = 0; i < k; ++i){
                ans.push_back('0'+i);
                string cur = ans.substr(ans.size()-n);
                if(visited.find(cur) == visited.end()){
                    visited.insert(cur);
                    if(backtrack(ans, visited)) return true;
                    visited.erase(cur);
                }
                ans.pop_back();
            }
            return false;
        }
    }
    
    string crackSafe(int n, int k) {
        this->n = n;
        this->k = k;
        total = pow(k, n);
        
        string ans(n, '0');
        unordered_set<string> visited = {ans};
        
        backtrack(ans, visited);
        
        return ans;
    }
};