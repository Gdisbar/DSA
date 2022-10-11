87. Scramble String
====================
// We can scramble a string s to get a string t using the following algorithm:

//     If the length of the string is 1, stop.
//     If the length of the string is > 1, do the following:
//         Split the string into two non-empty substrings at a random index, i.e., 
//         if the string is s, divide it to x and y where s = x + y.
//         Randomly decide to swap the two substrings or to keep them in the same 
//         order. i.e., after this step, s may become s = x + y or s = y + x.
//         Apply step 1 recursively on each of the two substrings x and y.

// Given two strings s1 and s2 of the same length, return true if s2 is a 
// scrambled string of s1, otherwise, return false.

 

// Example 1:

// Input: s1 = "great", s2 = "rgeat"
// Output: true
// Explanation: One possible scenario applied on s1 is:
// "great" --> "gr/eat" // divide at random index.
// "gr/eat" --> "gr/eat" 
// // random decision is not to swap the two substrings and keep them in order.
// "gr/eat" --> "g/r / e/at" 
// // apply the same algorithm recursively on both substrings. divide at random 
// //index each of them.
// "g/r / e/at" --> "r/g / e/at" 
// // random decision was to swap the first substring and to keep the second 
// //substring in the same order.
// "r/g / e/at" --> "r/g / e/ a/t" 
// // again apply the algorithm recursively, divide "at" to "a/t".
// "r/g / e/ a/t" --> "r/g / e/ a/t" 
// // random decision is to keep both substrings in the same order.
// The algorithm stops now, and the result string is "rgeat" which is s2.
// As one possible scenario led s1 to be scrambled to s2, we return true.

// Example 2:

// Input: s1 = "abcde", s2 = "caebd"
// Output: false

// Example 3:

// Input: s1 = "a", s2 = "a"
// Output: true

//Recursion only
class Solution {
    bool solve(string s1,string s2)
    {
        if(s1.size()==1)
            return s1==s2;
        if(s1==s2)
            return true;
        
        int n=s1.size();
        bool res = false;
        for(int i=1;i<n;++i)
        {
            if((solve(s1.substr(0,i),s2.substr(0,i)) and solve(s1.substr(i),s2.substr(i)))
               or(solve(s1.substr(0,i),s2.substr(n-i)) and solve(s1.substr(i),s2.substr(0,n-i))))
                return true;
        }
        return false;
    }


unordered_map<string,bool> mp;
bool f(string s1,string s2){
    int n=s1.size();
    if(s1==s2) return true;
    if(s1.size()==1) return s1==s2;
    string key=s1+s2;
    if(mp.find(key)!=mp.end()) return mp[key];
    vector<int>f1(26,0),f2(26,0);
    for(int i=0;i<n;++i){
        f1[s1[i]-'a']++;
        f2[s2[i]-'a']++;
    }
    if(f1!=f2) return mp[key]=false;
    for(int i=1;i<n;++i){
    	//ab|cdef --> x+y
    	//bc|daef or bcda|ef --> x'+y' or q+p
    	//x==x' && y==y' or x==p && y==q
        if((f(s1.substr(0,i),s2.substr(0,i))&&f(s1.substr(i),s2.substr(i))) or
          f(s1.substr(0,i),s2.substr(n-i))&&f(s1.substr(i),s2.substr(0,n-i)))
            return mp[key]=true;
    }
    return mp[key]=false;
}