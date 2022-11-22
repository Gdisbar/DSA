834. Sum of Distances in Tree
===================================
There is an undirected connected tree with n nodes labeled from 0 to n - 1 and n - 1 edges.

You are given the integer n and the array edges where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.

Return an array answer of length n where answer[i] is the sum of the distances between the ith node in the tree and all other nodes.

 

Example 1:

Input: n = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
Output: [8,12,6,10,10,10]
Explanation: The tree is shown above.
We can see that dist(0,1) + dist(0,2) + dist(0,3) + dist(0,4) + dist(0,5)
equals 1 + 1 + 2 + 2 + 2 = 8.
Hence, answer[0] = 8, and so on.

Example 2:

Input: n = 1, edges = []
Output: [0]

Example 3:

Input: n = 2, edges = [[1,0]]
Output: [1,1]


// <--- ret[root] = sum(count[child])+sum(count[child])
// should be
// <--- ret[root] = sum(res[child])+sum(count[child])

  # First DFS
         [TREE] |      [COUNT]     [RET]
            0   |       10          [ ] = (1+0) + (7+10) +(1+0)  = 19      
          / | \ |      / | \       / | \ 
         1  2  3|     1  7  1     0  10 0       <--- ret[root] = sum(count[child])+sum(count[child])
           /|\  |       /|\         /|\                        sum(count[child]) = travel again 「count[child]」 many times of path root->child 
          4 5 6 |      4 1 1       4 0 0                       sum(count[child]) = prev traveled paths sum
         /|     |     /|          /|   
        7 8     |    1 2         0 1    
         /      |     /           /         
        9       |    1           0           
        
   # Second DFS           
         [RET]  |                  |                  
            19  |           19     |       19             
          / | \ |          / | \   |      / | \ 
         0  10 0| [19-1+10-1] 10 0 |  28 [19-7+10-7] 0       <---  = parent.ret - root.count 
           /|\  |           /|\    |       /|\                      + (N - root.count)*1 
          4 0 0 |          4 0 0   |      4 0 0                       Eveny node other than it''s subtree node: become 1 step more far away
         /|     |         /|       |     /|   
        0 1     |        0 1       |    0 1    
         /      |         /        |     /         
        0       |        0         |    0         
        
    # Ans = [19,27,15,27,17,23,23,25,23,31] 
    --------------Example--------------------------------------------    
        
        Count         Ret         Dfs update ret...
          6            8            8              8     
         / \          / \          /  \           / \
        1   4        0   3    [8-1+N-1] 3       12   6
           /|\          /|\          /|\            /|\
          1 1 1        0 0 0        0 0 0        10 10 [6-1+N-1]   N=6

Explanation

    Let''s solve it with node 0 as root.

    Initial an array of hashset tree, tree[i] contains all connected nodes to i.
    Initial an array count, count[i] counts all nodes in the subtree i.
    Initial an array of res, res[i] counts sum of distance in subtree i.

    Post order dfs traversal, update count and res:
    count[root] = sum(count[i]) + 1
    res[root] = sum(res[i]) + sum(count[i])

    Pre order dfs traversal, update res:
    When we move our root from parent to its child i, count[i] points get 1 closer to root, n - count[i] nodes get 1 futhur to root.
    res[i] = res[root] - count[i] + N - count[i]

    return res, done.


vector<unordered_set<int>> tree;
    vector<int> res, count;

    vector<int> sumOfDistancesInTree(int N, vector<vector<int>>& edges) {
        tree.resize(N);
        res.assign(N, 0);
        count.assign(N, 1);
        for (auto e : edges) {
            tree[e[0]].insert(e[1]);
            tree[e[1]].insert(e[0]);
        }
        dfs(0, -1);
        dfs2(0, -1);
        return res;

    }

    void dfs(int root, int pre) {
        for (auto i : tree[root]) {
            if (i == pre) continue;
            dfs(i, root);
            count[root] += count[i];
            res[root] += res[i] + count[i];
        }
    }

    void dfs2(int root, int pre) {
        for (auto i : tree[root]) {
            if (i == pre) continue;
            res[i] = res[root] - count[i] + count.size() - count[i];
            dfs2(i, root);
        }
    }