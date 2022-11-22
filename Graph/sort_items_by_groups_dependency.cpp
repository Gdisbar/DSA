1203. Sort Items by Groups Respecting Dependencies
=====================================================
// There are n items each belonging to zero or one of m groups where group[i] 
// is the group that the i-th item belongs to and it''s equal to -1 if the i-th 
// item belongs to no group. The items and the groups are zero indexed. A group can 
// have no item belonging to it.

// Return a sorted list of the items such that:

//     The items that belong to the same group are next to each other in the 
//     sorted list.
//     There are some relations between these items where beforeItems[i] is a 
//     list containing all the items that should come before the i-th item in 
//     the sorted array (to the left of the i-th item).

// Return any solution if there is more than one solution and return an empty 
// list if there is no solution.

 

// Example 1:

// Input: n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1], 
// beforeItems = [[],[6],[5],[6],[3,6],[],[],[]]
// Output: [6,3,4,1,5,2,0,7]

// Example 2:

// Input: n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1], 
// beforeItems = [[],[6],[5],[6],[3],[],[4],[]]
// Output: []
// Explanation: This is the same as example 1 except that 4 needs to be before 
// 6 in the sorted list.


// I solved this problem first with two-level topological sort, but the code was 
// long and complicated. So I wanted to find a way to transform it to use a 
// generic topological sort.

//     Initialize adjacency list of n + 2m size. We will be using two extra nodes 
//     per group.
//     Wrap all nodes in the group with inbound and outbound dependency nodes.
//         Nodes not belonging to any group do not need to be wrapped.
//     Connect nodes directly for inter-group dependencies.
//     Connect to/from dependency nodes for intra-group dependencies.
//     Perform topological sort, starting from dependency nodes.
//         This way, all nodes within a single group will be together.
//     Remove dependency nodes from the result.

// In this example, the group dependency nodes are 8/10 (group 0) and 9/11 (group 1):

// The result of the topological sort could be, for example, 
// [0, 7, 8, 6, 3, 4, 10, 9, 2, 5, 11, 1], which after removing dependency nodes 
// becomes [0, 7, 6, 3, 4, 2, 5, 1].

bool topSort(vector<unordered_set<int>>& al, int i, vector<int>& res, 
		vector<int>& stat) {
    if (stat[i] != 0) return stat[i] == 2;
    stat[i] = 1;
    for (auto n : al[i])
        if (!topSort(al, n, res, stat)) return false;
    stat[i] = 2;
    res.push_back(i);
    return true;
}
vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
    vector<int> res_tmp, res(n), stat(n + 2 * m);
    vector<unordered_set<int>> al(n + 2 * m);
    for (auto i = 0; i < n; ++i) {
        if (group[i] != -1) {
            al[n + group[i]].insert(i);
            al[i].insert(n + m + group[i]);
        }
        for (auto j : beforeItems[i]) {
            if (group[i] != -1 && group[i] == group[j]) al[j].insert(i);
            else {
                auto ig = group[i] == -1 ? i : n + group[i];
                auto jg = group[j] == -1 ? j : n + m + group[j];
                al[jg].insert(ig);
            }
        }
    }
    for (int n = al.size() - 1; n >= 0; --n)
        if (!topSort(al, n, res_tmp, stat)) return {};
    reverse(begin(res_tmp), end(res_tmp));
    copy_if(begin(res_tmp), end(res_tmp), res.begin(), [&](int i) {return i < n; });
    return res;
}




// Simplify it with n+m adjacency list, with dfs topological sort.

class Solution {
public:
    vector<int> sortItems(int n, int m, vector<int>& group, 
    				vector<vector<int>>& beforeItems) {
        vector<vector<int>> graph(n + m);
        vector<int> indegree(n + m, 0);
        for(int i = 0;i < group.size();i++) {
            if(group[i] == -1) continue;
            graph[n+group[i]].push_back(i);
            indegree[i]++;
        }
        for(int i = 0;i < beforeItems.size();i++) {
            for(int e : beforeItems[i]) {
                int a = group[e] == -1 ? e : n + group[e];
                int b = group[i] == -1 ? i : n + group[i];
                if(a == b) { // same group, ingroup order
                    graph[e].push_back(i);
                    indegree[i]++;
                } else { // outgoup order
                    graph[a].push_back(b);
                    indegree[b]++;
                }
            }
        }
        vector<int> ans;
        for(int i = 0;i < n + m;i++) {
            if(indegree[i] == 0)
                dfs(ans, graph, indegree, n, i);
        }
        return ans.size() == n ? ans : vector<int>{};
    }
    
    void dfs(vector<int>& ans, vector<vector<int>>& graph, vector<int>& indegree, int n, int cur) {
        if(cur < n) ans.push_back(cur);
        indegree[cur] = -1; // mark it visited
        for(auto next : graph[cur]) {
            indegree[next]--;
            if(indegree[next] == 0)
                dfs(ans, graph, indegree, n, next);
        }
    }
};

