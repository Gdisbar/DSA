684. Redundant Connection
==============================
// In this problem, a tree is an undirected graph that is connected and has no cycles.

// You are given a graph that started as a tree with n nodes labeled from 1 to n, 
// with one additional edge added. The added edge has two different vertices chosen 
// from 1 to n, and was not an edge that already existed. The graph is represented 
// as an array edges of length n where edges[i] = [ai, bi] indicates that there is 
// an edge between nodes ai and bi in the graph.

// Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.

 

// Example 1:
// 										1
// 									  /   \
// 									 /-----\  
// 									2  	    3

// Input: edges = [[1,2],[1,3],[2,3]]
// Output: [2,3]

// Example 2:
// 								2------1-----5
// 								|      |
// 								|      |
// 								|      |
// 								3------4

// Input: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
// Output: [1,4]

class UnionFind {
    vector<int> parent, rank;
public:
    UnionFind(int n) {
        parent.resize(n); rank.resize(n);
        for (int i = 0; i < n; i++) {
            parent[i] = i; rank[i] = 1;
        }
    }
    int find(int x) {
        if (x == parent[x]) return x;
        return parent[x] = find(parent[x]); // Path compression
    }
    bool Union(int u, int v) {
        int pu = find(u), pv = find(v);
        if (pu == pv) return false; // Return False if u and v are already union
        if (rank[pu] > rank[pv]) { // Union by larger size
            rank[pu] += rank[pv];
            parent[pv] = pu;
        } else {
            rank[pv] += rank[pu];
            parent[pu] = pv;
        }
        return true;
    }
};

class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        UnionFind uf(n);
        for (auto& e : edges)
            if (!uf.Union(e[0]-1, e[1]-1)) return {e[0], e[1]};
        return {};
    }
};


// def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
//         self.already_connected = defaultdict(list)
//         for edge in edges:
//             self.visited = defaultdict(bool)
//             x, y = edge[0], edge[1]
//             if self.is_already_connected(x, y):
//                 return edge
//             self.already_connected[x].append(y)
//             self.already_connected[y].append(x)
            
//     def is_already_connected(self, x, y):
//         if x == y:
//             return True
//         for x_adjacent in self.already_connected[x]:
//             if not self.visited[x_adjacent]:
//                 self.visited[x_adjacent] = True
//                 if self.is_already_connected(x_adjacent, y):
//                     return True
//         return False


//dfs

// class Solution:
//     def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
//         self.already_connected = defaultdict(list)
//         for edge in edges:
//             self.visited = defaultdict(bool)
//             x, y = edge[0], edge[1]
//             if self.is_already_connected(x, y):
//                 return edge
//             self.already_connected[x].append(y)
//             self.already_connected[y].append(x)
            
//     def is_already_connected(self, x, y):
//         if x == y:
//             return True
//         for x_adjacent in self.already_connected[x]:
//             if not self.visited[x_adjacent]:
//                 self.visited[x_adjacent] = True
//                 if self.is_already_connected(x_adjacent, y):
//                     return True
//         return False


685. Redundant Connection II
===============================
// In this problem, a rooted tree is a directed graph such that, there is 
// exactly one node (the root) for which all other nodes are descendants of 
// this node, plus every node has exactly one parent, except for the root node 
// which has no parents.

// The resulting graph is given as a 2D-array of edges. Each element of edges is 
// a pair [ui, vi] that represents 
// a directed edge connecting nodes ui and vi, where ui is a parent of child vi.
// Return an edge that can be removed so that the resulting graph is a 
// rooted tree of n nodes. If there are multiple answers, return the answer 
// that occurs last in the given 2D-array.


// Example 1:

// Input: edges = [[1,2],[1,3],[2,3]]
// Output: [2,3]

// Example 2:

// Input: edges = [[1,2],[2,3],[3,4],[4,1],[1,5]]
// Output: [4,1]


//DSU
/*
We can remove only one node!
There are three cases for the tree structure to be invalid.
1) A node having two parents; lets call it special node(having that imposter edge)
   including corner case: e.g. [[1,2],[1,3],[2,3]]
2) A cycle exists , but no candidates     [[1,2],[2,3],[3,4],[4,1],[1,5]]
3) A cycle exists, and canidates exist(special node)   [[2,1],[3,1],[4,2],[1,4]]

Either cycle exists or special node occurs(with or without cycle)   //ples dont get confused by this line, maybe just ignore it



Process:
1) Check whether there is a node having two parents.      case 1 
    If so,  store them as candidates A and B.  (either both can occurs or none of them)
    candA occurs before candB
    
2)  Then we need to look for cycles
     if no candidates then                                  case 2


    if candidates exist                                     case 3
    


3) Then we make unions of all nodes except our special node.
SO, first of all, If at any point we find a cycle, then our first priority is to look for candA(because it is potential imposter and it takes part in union process)             case 3

TRY TO SIMULATE PROCESS IN [[2,1],[3,1],[4,2],[1,4]]  for more info why candA will go first..
If candA exists, it means that it is the one that takes part in cycle formation and 2 parent situatin, so we remove it!

If candA doesnot exist (neither candA nor candB), then it means current edge is imposter,
that leads to cycle formation, same as we did in REDUNDANT CONNECTION I (https://leetcode.com/problems/redundant-connection/)
return current edge                         case 2

Now , we have seen all cases of possible cycle formation. It means we have made all unions of nodes except our special edge and  we did not found any cycles.
So it means it the situation of 2 parent that leads to invalid tree. But this time we have candA and candB (both) as potential imposters....which one to choose...
Either of them can be removed and we will get a valid tree.
TRY TO SIMULATE IN [[2,1],[3,1],[4,3],[1,4]] for more info why candB will go.    PS(it is a different test case but similar to one mentioned above)..

But according to ques, we want one that occurs last, so its candB!             case 1

We covered all cases..So, hope it was clear!
Just try to simulate the testcases given in beginning with the following code. 

*/


class Solution {
public:
    
    class UnionFind {
        public:  
            vector<int> parent;
            int count = 0;
            UnionFind(int n){                  //constructor
                count = n;
                parent = vector<int>(n+1,-1);
            }

            int find(int x){
                if(parent[x]==-1) return x;
                return find(parent[x]);
            }

            bool Union(int x,int y){
                int X = find(x);
                int Y = find(y);
                cout<<X<<" "<<Y<<endl;
                if(X==Y) return true;   //cycle found

                parent[Y]=X;         //A ->B
                count--;
                return false;
            }
            

            int getCount(){
                return count;
            }
    };
    
    
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
        int n = size(edges);
        vector<int> parent(n+1, 0), candA, candB;
        // step 1, check whether there is a node with two parents
        //candA occured before candB
        
        for (auto &edge:edges) {
            if (parent[edge[1]] == 0)
                parent[edge[1]] = edge[0]; 
            else {
                candA = {parent[edge[1]], edge[1]};
                candB = edge;
                edge[1] = 0;     //the edge(canB) destination is marked as 0 to identify it uniquely , the node with 2 parents..
                //the edge candA will still take part in union process..
                
            }
        } 
        
        
        
        UnionFind uf(n);
        
        
        for(auto& edge : edges) {
            if (edge[1] == 0) continue;    //wait for now, dont make connections from special node now
            int u = edge[0], v = edge[1];
           
            if(uf.Union(u,v)){
                //returned true, means cycle exists
                if(candA.size()){
                    cout<<"A";
                    return candA;
                }
                else{
                    cout<<"edge";
                    return edge;
                }
                
            }
        }
        
        cout<<"B";
        return candB;
    
    }
        
};

[[1,2],[1,3],[2,3]]
[[1,2],[2,3],[3,4],[4,1],[1,5]]
[[1,2],[3,1],[2,3]]
1 2
1 3
B1 2
1 3
1 4
1 1
edge1 2
3 1
3 3
edge
Output
[2,3]
[4,1]

class Solution {
public:
    vector<int> root;
    
    int Find(int x) {
        if (root[x] < 0) return x;
        return root[x] = Find(root[x]);
    }
    
    bool Union(int x, int y) {
        int rx = Find(x);
        int ry = Find(y);
        if (rx == ry) return 1;
        if (root[rx] > root[ry]) {
            swap(rx, ry);
        }
        root[rx] += root[ry];
        root[ry] = rx;
        return 0;
    }
    
    vector<int> findRedundantDirectedConnection(vector<vector<int>> &edges) {
        
        int a, b, v = -1;
        root.assign(edges.size()+1, 0);
        
        for (auto &e : edges) {
            if (root[e[1]]) {
                a = root[e[1]];
                b = e[0];
                v = e[1];
                break;
            }
            root[e[1]] = e[0];
        }
        
        root.assign(edges.size()+1, -1);
        
        if (v == -1) {
            for (auto &e : edges) {
                if (Union(e[0], e[1])) {
                    return e;
                }
            }
        } else {
            for (auto &e : edges) {
                if (e[0] == b) continue;
                if (Union(e[0], e[1])) {
                    return {a, v};
                }
            }
            return {b, v};
        }
        
        return {};
    }
};