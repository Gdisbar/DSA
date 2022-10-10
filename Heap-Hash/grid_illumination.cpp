1001. Grid Illumination
=========================
// There is a 2D grid of size n x n where each cell of this grid has a lamp that is 
// initially turned off.

// You are given a 2D array of lamp positions lamps, where lamps[i] = [rowi, coli] 
// indicates that the lamp at grid[rowi][coli] is turned on. Even if the same lamp is 
// listed more than once, it is turned on.

// When a lamp is turned on, it illuminates its cell and all other cells in the same 
// row, column, or diagonal.

// You are also given another 2D array queries, where queries[j] = [rowj, colj]. 
// For the jth query, determine whether grid[rowj][colj] is illuminated or not. 
// After answering the jth query, turn off the lamp at grid[rowj][colj] and its 8 
// adjacent lamps if they exist. A lamp is adjacent if its cell shares either a side 
// or corner with grid[rowj][colj].

// Return an array of integers ans, where ans[j] should be 1 if the cell in the jth 
// query was illuminated, or 0 if the lamp was not.

 

// Example 1:

// Input: n = 5, lamps = [[0,0],[4,4]], queries = [[1,1],[1,0]]
// Output: [1,0]
// Explanation: We have the initial grid with all lamps turned off. In the above 
// picture we see the grid after turning on the lamp at grid[0][0] then turning on the 
// lamp at grid[4][4].
// The 0th query asks if the lamp at grid[1][1] is illuminated or not (the blue square). 
// It is illuminated, so set ans[0] = 1. Then, we turn off all lamps in the red square.

// The 1st query asks if the lamp at grid[1][0] is illuminated or not (the blue square). 
// It is not illuminated, so set ans[1] = 0. Then, we turn off all lamps in the red 
// rectangle.

// Example 2:

// Input: n = 5, lamps = [[0,0],[4,4]], queries = [[1,1],[1,1]]
// Output: [1,1]

// Example 3:

// Input: n = 5, lamps = [[0,0],[0,4]], queries = [[0,4],[0,1],[1,4]]
// Output: [1,1,0]

// duplicates
// Input:
// 6
// [[2, 5], [4, 2], [0, 3], [0, 5], [1, 4], [4, 2], [3, 3], [1, 0]]
// [[4, 3], [3, 1], [5, 3], [0, 5], [4, 4], [3, 3]]
// Output:
// [1,1,1,1,1,1]
// Expected:
// [1,0,1,1,0,1]

class Solution {
public:
    struct pairHash {
            size_t operator()(const pair<int, int> &x) const { return x.first ^ x.second; }
    };
    vector<int> gridIllumination(int n, vector<vector<int>>& lamps, vector<vector<int>>& queries) {
       vector<int> ans;
       unordered_map<int,int> x,y,p_diag,s_diag;
       //unordered_map<int,set<int>> s;
       unordered_set<pair<int,int>,pairHash> s; //track lamp positions
       for(auto lmp : lamps){
           int i = lmp[0],j=lmp[1];
           if(s.insert({i,j}).second){ //s[i].insert(j).second //duplicate check , 
            //1st element = inserted position , 
            //2nd element = true if element inserted false if element already present
               x[i]++,y[j]++,p_diag[i+j]++,s_diag[i-j]++;
           }
       }
      for(auto q : queries){
          int i = q[0],j=q[1];
          if(x[i]>0||y[j]>0||p_diag[i+j]>0||s_diag[i-j]>0){
              ans.push_back(1);
              for(int ii=i-1;ii<=i+1;++ii){
                  for(int jj=j-1;jj<=j+1;++jj){
                      if(s.erase({ii,jj})){ //s[ii].erase(jj)
                          x[ii]--,y[jj]--,p_diag[ii+jj]--,s_diag[ii-jj]--;
                      }
                  }
              }
          }
          else{
              ans.push_back(0);
          }
      }
        return ans;
    }
};