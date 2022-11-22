797. All Paths From Source to Target
=====================================
// Given a directed acyclic graph (DAG) of n nodes labeled from 0 to n - 1, 
// find all possible paths from node 0 to node n - 1 and return them in any order.

// The graph is given as follows: graph[i] is a list of all nodes you can visit 
// from node i (i.e., there is a directed edge from node i to node graph[i][j]).

 

// Example 1:

// Input: graph = [[1,2],[3],[3],[]]
// Output: [[0,1,3],[0,2,3]]
// Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.

// Example 2:

// Input: graph = [[4,3,1],[3,2,4],[3],[4],[]]
// Output: [[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]

 

// Constraints:

//     n == graph.length
//     2 <= n <= 15
//     0 <= graph[i][j] < n
//     graph[i][j] != i (i.e., there will be no self-loops).
//     All the elements of graph[i] are unique.
//     The input graph is guaranteed to be a DAG.

// If it asks just the number of paths, generally we can solve it in two ways.

//     Count from start to target in topological order.
//     Count by dfs with memo.
//     Both of them have time O(Edges) and O(Nodes) space. Let me know if you 
//     agree here.

// I didn''t do that in this problem, for the reason that it asks all paths. I don''t 
// expect memo to save much time. (I didn''t test).
// Imagine the worst case that we have node-1 to node-N, and node-i linked to 
// node-j if i < j.
// There are 2^(N-2) paths and (N+2)*2^(N-3) nodes in all paths. We can roughly 
// say O(2^N).


void dfs(vector<vector<int>>& g, vector<vector<int>>& res, 
								vector<int>& path, int cur) {
        path.push_back(cur);
        if (cur == g.size() - 1)
            res.push_back(path);
        else for (auto it: g[cur])
            dfs(g, res, path, it);
        path.pop_back();

    }

    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& g) {
        vector<vector<int>> paths;
        vector<int> path;
        dfs(g, paths, path, 0);
        return paths;
    }


def allPathsSourceTarget(self, graph):
    def dfs(cur, path):
        if cur == len(graph) - 1: res.append(path)
        else:
            for i in graph[cur]: dfs(i, path + [i])
    res = []
    dfs(0, [0])
    return res