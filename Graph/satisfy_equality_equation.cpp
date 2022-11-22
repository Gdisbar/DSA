990. Satisfiability of Equality Equations
==============================================
// You are given an array of strings equations that represent relationships 
// between variables where each string equations[i] is of length 4 and takes one of 
// two different forms: "xi==yi" or "xi!=yi".Here, xi and yi are lowercase 
// letters (not necessarily different) that represent one-letter variable names.

// Return true if it is possible to assign integers to variable names so as to 
// satisfy all the given equations, or false otherwise.

 

// Example 1:

// Input: equations = ["a==b","b!=a"]
// Output: false
// Explanation: If we assign say, a = 1 and b = 1, then the first equation is 
// satisfied, but not the second.
// There is no way to assign the variables to satisfy both equations.

// Example 2:

// Input: equations = ["b==a","a==b"]
// Output: true
// Explanation: We could assign a = 1 and b = 1 to satisfy both equations.


class Solution {
public:
    vector<int> parent;
    int findParent(int i){
        if(parent[i] == -1) return i;
        return parent[i] = findParent(parent[i]);
    }
    
    void unionNode(int a, int b){
        int p1 = findParent(a);
        int p2 = findParent(b);
        if(p1 != p2) parent[p1] = p2;
    }
    bool equationsPossible(vector<string>& equations) {
        parent.resize(26, -1); //as all are alphabets max possible nodes can be 26
        for(auto e:equations){
            if(e[1] == '=') 
                unionNode(e[0]-'a', e[3]-'a');
        }
        for(auto e:equations){
            if(e[1] == '!' && findParent(e[0]-'a') == findParent(e[3]-'a') ) 
                return false;
        }
        return true;
    }
};