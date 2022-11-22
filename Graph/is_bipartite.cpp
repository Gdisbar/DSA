785. Is Graph Bipartite?
=============================
There is an undirected graph with n nodes, where each node is numbered between 0 and n - 1. You are given a 2D array graph, where graph[u] is an array of nodes that node u is adjacent to. More formally, for each v in graph[u], there is an undirected edge between node u and node v. The graph has the following properties:

    There are no self-edges (graph[u] does not contain u).
    There are no parallel edges (graph[u] does not contain duplicate values).
    If v is in graph[u], then u is in graph[v] (the graph is undirected).
    The graph may not be connected, meaning there may be two nodes u and v such that there is no path between them.

A graph is bipartite if the nodes can be partitioned into two independent sets A and B such that every edge in the graph connects a node in set A and a node in set B.

Return true if and only if it is bipartite.

 

Example 1:

Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
Explanation: There is no way to partition the nodes into two independent sets such that every edge connects a node in one and a node in the other.

Example 2:

Input: graph = [[1,3],[0,2],[1,3],[0,2]]
Output: true
Explanation: We can partition the nodes into two sets: {0, 2} and {1, 3}.

// Time    O(V+E)
// Space   O(V+E)

Our goal is trying to use two colors to color the graph and see if there are any adjacent nodes having the same color.
Initialize a color[] array for each node. Here are three states for colors[] array:
0: Haven''t been colored yet.
1: Blue.
-1: Red.
For each node,

    If it hasn''t been colored, use a color to color it. Then use the other color to color all its adjacent nodes (DFS).
    If it has been colored, check if the current color is the same as the color that is going to be used to color it. (Please forgive my english... Hope you can understand it.)


class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size();
        vector<int> colors(n, 0);
        for (int i = 0; i < n; i++) {
            if (!colors[i] && !valid(graph, colors, i, 1)) {
                return false;
            }
        }
        return true;
    }
private:
    bool valid(vector<vector<int>>& graph, vector<int>& colors, int node, int color) {
        if (colors[node]) {
            return colors[node] == color;
        }
        colors[node] = color;
        for (int neigh : graph[node]) {
            if (!valid(graph, colors, neigh, -color)) {
                return false;
            }
        }
        return true;
    }
};

To be able to split the node set {0, 1, 2, ..., (n-1)} into sets A and B, we will try to color nodes in set A with color A (i.e., value 1) and nodes in set B with color B (i.e., value -1), respectively.

If so, the graph is bipartite if and only if the two ends of each edge must have opposite colors. Therefore, we could just start with standard BFS to traverse the entire graph and

    color neighbors with opposite color if not colored, yet;
    ignore neighbors already colored with oppsite color;
    annouce the graph can''t be bipartite if any neighbor is already colored with the same color.

NOTE: The given graph might not be connected, so we will need to loop over all nodes before BFS.



// bfs , V+E
bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> color(n); // 0: uncolored; 1: color A; -1: color B
        
    queue<int> q; // queue, resusable for BFS    
    
    for (int i = 0; i < n; i++) {
      if (color[i]) continue; // skip already colored nodes
      
      // BFS with seed node i to color neighbors with opposite color
      color[i] = 1; // color seed i to be A (doesn't matter A or B) 
      for (q.push(i); !q.empty(); q.pop()) {
        int cur = q.front();
        for (int neighbor : graph[cur]) 
        {
          if (!color[neighbor]) // if uncolored, color with opposite color
          { color[neighbor] = -color[cur]; q.push(neighbor); } 
          
          else if (color[neighbor] == color[cur]) 
            return false; // if already colored with same color, can't be bipartite!
        }        
      }
    }
    
    return true;
  }


he key information in the problem description is that **edges are only for bridging between the two sets. **

    A node and any of its neightbor should NOT be in the same set => this is easy to understand, since node and its neighbor forms an edge, and an edge cross the two sets.
    All of a node's neighbors should be in the same set => since there are only two sets, if there are one set which contains the node itself, then the neighbors should belong to the other set.

It's natural to use UnionFind to represent sets. The code just loop through all the node and its neighbors and make sure the above two conditions are met. Time complexity should be O(V + 2E) while V is the total number of nodes and E is the total number of edges in the graph.

class UnionFind {
public:
    UnionFind(int size): size(size), parent(size), rank(size, 0) {  
        for(int i = 0; i < size; i++) {
            parent[i] = i;
        }
    }
    
    bool Union(int x, int y) {
        int px = find(x);
        int py = find(y);
        if(px == py)
            return false;
        if(rank[px] < rank[py]) {
            parent[px] = py;
            rank[py]++;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        size--;
        return true;
    }
    
    int find(int x) {
        while(parent[x] != parent[parent[x]]) {
            parent[x] = find(parent[x]);
        }
        
        return parent[x];
    }
    
    vector<int> parent;
    vector<int> rank;
    int size;
};

class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph) {
        // Union Find
        const int n = graph.size();
        UnionFind uf(n);
        
        for(int i = 0; i < n; i++) {
            if(graph[i].empty()) {
                continue;
            }
            for(int j : graph[i]) {
                // an edge should cross the two sets
                if(uf.find(i) == uf.find(j)) {
                    return false;
                }
            }
            // all neighbors of one element should be in same sets
            int node = graph[i][0];
            for(int j = 1; j < graph[i].size(); j++) {
                uf.Union(node, graph[i][j]);
            }
        }
        
        return true;
    }
};
