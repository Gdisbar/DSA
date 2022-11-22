1202. Smallest String With Swaps
====================================
// You are given a string s, and an array of pairs of indices in the string 
// pairs where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.

// You can swap the characters at any pair of indices in the given pairs any 
// number of times.

// Return the lexicographically smallest string that s can be changed to after 
// using the swaps.

 

// Example 1:

// Input: s = "dcab", pairs = [[0,3],[1,2]]
// Output: "bacd"
// Explaination: 
// Swap s[0] and s[3], s = "bcad"
// Swap s[1] and s[2], s = "bacd"

// Example 2:

// Input: s = "dcab", pairs = [[0,3],[1,2],[0,2]]
// Output: "abcd"
// Explaination: 
// Swap s[0] and s[3], s = "bcad"
// Swap s[0] and s[2], s = "acbd"
// Swap s[1] and s[2], s = "abcd"

// Example 3:

// Input: s = "cba", pairs = [[0,1],[1,2]]
// Output: "abc"
// Explaination: 
// Swap s[0] and s[1], s = "bca"
// Swap s[1] and s[2], s = "bac"
// Swap s[0] and s[1], s = "abc"


class Solution {
private:
    vector<int> ind;                                              
    vector<bool> vis;
    vector<vector<int>> adj;
    string ss;                                                 
    void dfs(string &s,int i){
        vis[i]=true;
        ind.push_back(i);
        ss+=s[i];
        for(auto it:adj[i])
            if(!vis[it])
               dfs(s,it);
    }
public:
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
         adj.resize(s.size());
        vis.resize(s.size(),false);
        for(auto p:pairs){
            adj[p[0]].push_back(p[1]);
            adj[p[1]].push_back(p[0]);
        }                          
            
        for(int i=0;i<s.size();++i)
            if(!vis[i]){
                ss="";                              
                ind.clear();                            
                dfs(s,i);
                sort(ss.begin(),ss.end());                   
                sort(ind.begin(),ind.end());                                 
                for(int i=0;i<ind.size();i++)         
                    s[ind[i]]=ss[i];
            }
        return s;
    }
};