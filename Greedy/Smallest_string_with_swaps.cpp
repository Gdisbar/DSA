1202. Smallest String With Swaps
==================================
// You are given a string s, and an array of pairs of indices in the string pairs 
// where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.

// You can swap the characters at any pair of indices in the given pairs any number 
// of times.

// Return the lexicographically smallest string that s can be changed to after using 
// the swaps.

 

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

// Observation
// When there are multiple indices that overlap each other eg: [1,2] and [2,3] 
// we can always get the characters at those indices (1,2,3) at any of the indice 
// we like by swapping in some permutation.
// These overlapped indices form our "connected graph or connected components" and 
// these belong to one group.
// eg: [2,3],[4,5],[3,6],[2,7] for a string of length 8 we have the following groups:

//     0
//     1
//     2,3,6,7
//     4,5

// We use this observation to build our solution.

// Solution
// All we need to do is get all those groups of indices that can be swapped 
// with each other and sort the string formed by those indices since they can always 
// be replaced/swapped by any other charater in those indices as noted above.
// Repeat this for all the groups and you get your sorted string.



class Solution {
public:
    vector<int> indices;   //Stores indices of same group.
    vector<bool> visited;
    vector<vector<int>> adjList;
    string indiceString;       //Stores  string formed by indices in the same group.
    void dfs(string &s,int n)    //DFS to get all indices in same group.
    {
        visited[n]=true;
        indices.push_back(n);
        indiceString+=s[n];
        for(int &i:adjList[n])
            if(!visited[i])
               dfs(s,i);
    }
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) 
    {
        adjList.resize(s.length());
        visited.resize(s.length(),false);
        for(vector<int> &v:pairs)  //Create adjacency list using the indice pairs
            adjList[v[0]].push_back(v[1]),adjList[v[1]].push_back(v[0]);
        for(int i=0;i<s.length();i++)
            if(!visited[i]){
                indiceString="";  //Clear string formed by one group of indices before finding next group.
                indices.clear();   //Clear indices vector before finding another group.
                dfs(s,i);
                sort(indiceString.begin(),indiceString.end());//Sort the characters in the same group.
                sort(indices.begin(),indices.end());  //Sort the indices in the same group.            
                for(int i=0;i<indices.size();i++)  //Replace all the indices in the same group with the sorted characters.
                    s[indices[i]]=indiceString[i];
            }
        return s;
    }
};


// Disjoint Set

int find(vector<int>& ds, int i) {
  return ds[i] < 0 ? i : ds[i] = find(ds, ds[i]);
}
string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
  vector<int> ds(s.size(), -1);
  vector<vector<int>> m(s.size());
  for (auto& p : pairs) {
    auto i = find(ds, p[0]), j = find(ds, p[1]);
    if (i != j) {
        if (-ds[i] < -ds[j]) 
            swap(i, j);
        ds[i] += ds[j];
        ds[j] = i;
    }
  }
  for (auto i = 0; i < s.size(); ++i) 
      m[find(ds, i)].push_back(i);
  for (auto &ids : m) {
    string ss = "";
    for (auto id : ids) 
        ss += s[id];
    sort(begin(ss), end(ss));
    for (auto i = 0; i < ids.size(); ++i) 
        s[ids[i]] = ss[i];
  }
  return s;
}

Choose and Swap
===================
// You are given a string s of lower case english alphabets. 
// You can choose any two characters in the string and replace all the occurences 
// of the first character with the second character and replace all the occurences 
// of the second character with the first character. Your aim is to find the 
// lexicographically smallest string that can be obtained by doing this operation 
// at most once.

// Example 1:

// Input:
// A = "ccad"
// Output: "aacd"
// Explanation:
// In ccad, we choose a and c and after 
// doing the replacement operation once we get, 
// aacd and this is the lexicographically
// smallest string possible. 


// Function to return the lexicographically
// smallest string after swapping all the
// occurrences of any two characters
string smallestStr(string str, int n)
{
    int i, j;
    // To store the first index of
    // every character of str
    int chk[256]={-1};
   
    // Store the first occurring
    // index every character
    for (i = 0; i < n; i++) {
   
        // If current character is appearing for the first time in str
        if (chk[str[i] - 'a'] == -1)
            chk[str[i] - 'a'] = i;
    }
   
    // Starting from the leftmost character
    for (i = 0; i < n; i++) {
   
        bool flag = false;
   
        // For every character smaller than str[i]
        for (j = 0; j < str[i] - 'a'; j++) {
   
            // If there is a character in str which is
            // smaller than str[i] and appears after it
            if (chk[j] > chk[str[i] - 'a']) {
                flag = true;
                break;
            }
        }
   
        // If the required character pair is found
        if (flag)
            break;
    }
   
    // If swapping is possible
    if (i < n) {
   
        // Characters to be swapped
        char ch1 = str[i];
        char ch2 = char(j + 'a');
   
        // For every character
        for (i = 0; i < n; i++) {
   
            // Replace every ch1 with ch2
            // and every ch2 with ch1
            if (str[i] == ch1)
                str[i] = ch2;
   
            else if (str[i] == ch2)
                str[i] = ch1;
        }
    }
   
    return str;
}