1857. Largest Color Value in a Directed Graph
===================================================

// There is a directed graph of n colored nodes and m edges. The nodes are numbered 
// from 0 to n - 1.

// You are given a string colors where colors[i] is a lowercase English letter 
// representing the color of the ith node in this graph (0-indexed). You are also 
// given a 2D array edges where edges[j] = [aj, bj] indicates that there is a 
// directed edge from node aj to node bj.

// A valid path in the graph is a sequence of nodes x1 -> x2 -> x3 -> ... -> xk 
// such that there is a directed edge from xi to xi+1 for every 1 <= i < k. The 
// color value of the path is the number of nodes that are colored the most 
// frequently occurring color along that path.

// Return the largest color value of any valid path in the given graph, or -1 if 
// the graph contains a cycle.

 

// Example 1:
//                 a     a     c     a
//                 0 --> 2 --> 3 --> 4 
//                 |
//               b 1

// Input: colors = "abaca", edges = [[0,1],[0,2],[2,3],[3,4]]
// Output: 3
// Explanation: The path 0 -> 2 -> 3 -> 4 contains 3 nodes that are colored "a" 

// Example 2:

// Input: colors = "a", edges = [[0,0]]
// Output: -1
// Explanation: There is a cycle from 0 to 0.


// Intuition: We can use BFS Topological Sort to visit the nodes. 
// When visiting the next node, we can forward the color information to the next 
// node. Also Topo-sort can help detect circle.

// Algorithm:

// Just do normal topo sort. One modification is that, for each node, we need to 
// store a int cnt[26] array where cnt[i] is the maximum count of color i in all 
// paths to the current node.

// For example, assume there are two paths reaching the current node, aba, bba. 
// Then cnt['a'] = 2 and cnt['b'] = 2 because both color a and b can be 2 in 
// different paths.


int largestPathValue(string c, vector<vector<int>>& edges) {
        unordered_map<int,vector<int>> adj;
        // cnt[i][j] is the maximum count of j-th color from the ancester nodes to 
        //node i.
        vector<vector<int>> cnt(c.size(),vector<int>(26));
        vector<int> indeg(c.size(),0);
        for(auto e : edges){
            adj[e[0]].push_back(e[1]);
            indeg[e[1]]++;
        }
        queue<int> q;
        int ans=0,seen=0;
        for(int i=0;i<c.size();++i)
                if(indeg[i]==0){  // indegree 0, we can use it as a source node
                     q.push(i);
                     // the count of the current color should be 1
                    cnt[i][c[i]-'a']=1;  
                }
        while(!q.empty()){
            auto u = q.front();
            q.pop();
             // we use the maximum of all the maximum color counts as the 
            // color value.
            int val = *max_element(begin(cnt[u]),end(cnt[u]));
            ans=max(ans,val);
            ++seen;
            for(auto v : adj[u]){
                for(int i=0;i<26;++i){
                    // try to use node `u` to update all the color counts of 
                    //node `v`.
                    cnt[v][i] = max(cnt[v][i], cnt[u][i] + (i == c[v] - 'a'));
                }
                if(--indeg[v]==0)
                q.push(v);
            }
            
        }
        
        return seen < c.size() ? -1 : ans;
    }