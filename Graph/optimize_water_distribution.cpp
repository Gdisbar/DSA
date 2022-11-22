1168 - Optimize Water Distribution in a Village
===================================================

// There are n houses in a village. We want to supply water for all the houses by 
// building wells and laying pipes.

// For each house i, we can either build a well inside it directly with cost 
// wells[i], or pipe in water from another well to it. The costs to lay pipes 
// between houses are given by the array pipes, where each 
// pipes[i] = [house1, house2, cost] represents the cost to connect house1 and 
// house2 together using a pipe. Connections are bidirectional.

// Find the minimum total cost to supply water to all houses.

// Example 1: 
                            //   1    1
                            // 1----2----3
                             
                               
// Input: n = 3, wells = [1,2,2], pipes = [[1,2,1],[2,3,1]]
// Output: 3
// Explanation: 
// The image shows the costs of connecting houses using pipes.
// The best strategy is to build a well in the first house with cost 1 
// and connect the other houses to it with cost 2 so the total cost is 3.


// we can add extra node as root node 0, and connect all 0 and i with 
// costs wells[i]. so we can transform to a spanning tree 

              //  (3)  2  (9)  4  (12)     
              //   1-------2-------3   
              //    \     /\ 14   / 
              // 12  \   /  \    / 7
              //      \ / 8  \  /
              // (10)  4       5 (6)

.
// Complexity Analysis

    // Time Complexity: O(ElogE) - E number of edge in graph
    // Space Complexity: O(E)

    // A graph at most have n(n-1)/2 - n number of nodes in graph edges 
    // （Complete Graph）


//     Build graph with all possible edges.
//     Sort edges by value (costs)
//     Iterate all edges (from min value to max value)
//     For each edges, check whether two nodes already connected (union-find),
//         if already connected, then skip
//         if not connected, then union two nodes, add costs to result


class UnionFind{
private:
    vector<int> unionFind;
public:
    UnionFind(int N):unionFind(N){
        for(int i=0;i<N;i++) unionFind[i]=i;
    }
    void merge(int key1,int key2){
        key1=get(key1);
        key2=get(key2);
        if(key1==key2) return;
        unionFind[key2]=key1;
    }
    int get(int key){
        while(key!=unionFind[key])
            key=unionFind[key]=unionFind[unionFind[key]];
        return key;
    }
};
class Solution {
public:
    int minCostToSupplyWater(int n, vector<int>& wells, 
                vector<vector<int>>& pipes) {
        for(int i=1;i<=n;i++)
            pipes.push_back({0,i,wells[i-1]});
        // Kruskal's algorithm
        UnionFind uf(n+1);
        sort(pipes.begin(),pipes.end(),[](auto& a,auto& b){
            return b[2]>a[2]; 
        });
        int res=0;
        for(auto& pipe:pipes){
            if(uf.get(pipe[0])!=uf.get(pipe[1])){
                res+=pipe[2];
                uf.merge(pipe[0],pipe[1]);
            }
        }
        return res;
    }
};