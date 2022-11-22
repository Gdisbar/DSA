1579. Remove Max Number of Edges to Keep Graph Fully Traversable
==================================================================
// Alice and Bob have an undirected graph of n nodes and three types of edges:

//     Type 1: Can be traversed by Alice only.
//     Type 2: Can be traversed by Bob only.
//     Type 3: Can be traversed by both Alice and Bob.

// Given an array edges where edges[i] = [typei, ui, vi] represents a 
// bidirectional edge of type typei between nodes ui and vi, find the maximum 
// number of edges you can remove so that after removing the edges, the graph 
// can still be fully traversed by both Alice and Bob. The graph is fully 
// traversed by Alice and Bob if starting from any node, they can reach all 
// other nodes.

// Return the maximum number of edges you can remove, or return -1 if Alice and 
// Bob cannot fully traverse the graph.

 

// Example 1:

// Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]]
// Output: 2
// Explanation: If we remove the 2 edges [1,1,2] and [1,1,3]. The graph will 
// still be fully traversable by Alice and Bob. Removing any additional edge 
// will not make it so. So the maximum number of edges we can remove is 2.

// Example 2:

// Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,4],[2,1,4]]
// Output: 0
// Explanation: Notice that removing any edge will not make the graph fully 
// traversable by Alice and Bob.

// Example 3:

// Input: n = 4, edges = [[3,2,3],[1,1,2],[2,3,4]]
// Output: -1
// Explanation: In the current graph, Alice cannot reach node 4 from the other 
// nodes. Likewise, Bob cannot reach 1. Therefore it''s impossible to make the 
// graph fully traversable.

// The idea here is to think that initially the graph is empty and now we want to 
// add the edges into the graph such that graph is connected.

// Union-Find is an easiest way to solve such problem where we start with all 
// nodes in separate components and merge the nodes as we add edges into the graph.

// As some edges are available to only Bob while some are available only to Alice, 
// we will have two different union find objects to take care of their own 
// traversability.

// Key thing to remember is that we should prioritize type 3 edges over type 
// 1 and 2 because they help both of them at the same time.

class UnionFind {
    vector<int> component;
    int distinctComponents;
public:
    /*
     *   Initially all 'n' nodes are in different components.
     *   e.g. component[2] = 2 i.e. node 2 belong to component 2.
     */
    UnionFind(int n) {
	    distinctComponents = n;
        for (int i=0; i<=n; i++) {
            component.push_back(i);
        }
    }
    
    /*
     *   Returns true when two nodes 'a' and 'b' are initially in different
     *   components. Otherwise returns false.
     */
    bool unite(int a, int b) {       
        if (findComponent(a) == findComponent(b)) {
            return false;
        }
        component[findComponent(a)] = b;
        distinctComponents--;
        return true;
    }
    
    /*
     *   Returns what component does the node 'a' belong to.
     */
    int findComponent(int a) {
        if (component[a] != a) {
            component[a] = findComponent(component[a]);
        }
        return component[a];
    }
    
    /*
     *   Are all nodes united into a single component?
     */
    bool united() {
        return distinctComponents == 1;
    }
};



// ----------------- Actual Solution --------------
class Solution {
    
public:
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        // Sort edges by their type such that all type 3 edges will be at the beginning.
        sort(edges.begin(), edges.end(), [] (vector<int> &a, vector<int> &b) { return a[0] > b[0]; });
        
        int edgesAdded = 0; // Stores the number of edges added to the initial empty graph.
        
        UnionFind bob(n), alice(n); // Track whether bob and alice can traverse the entire graph,
                                    // are there still more than one distinct components, etc.
        
        for (auto &edge: edges) { // For each edge -
            int type = edge[0], one = edge[1], two = edge[2];
            switch(type) {
                case 3:
                    edgesAdded += (bob.unite(one, two) | alice.unite(one, two));
                    break;
                case 2:
                    edgesAdded += bob.unite(one, two);
                    break;
                case 1:
                    edgesAdded += alice.unite(one, two);
                    break;
            }
        }
        
        return (bob.united() && alice.united()) ? (edges.size()-edgesAdded) : -1; // Yay, solved.
    }
};

// faster

class Solution {
    
    int find(int p,vector<int>&parent){
        if(parent[p]==p) return p;
        return parent[p]=find(parent[p],parent);
    }
    bool union_(int u,int v,vector<int>&parent,vector<int>&rank){
        int x= find(u,parent);
        int y=find(v,parent);
        if(x!=y){
            if(rank[x]>rank[y]){
                parent[y]=x;
                //rank[x]++;
            }
            else if(rank[x]<rank[y]){
                parent[x]=y;
                //rank[y]++;
            }
            else{
                parent[x]=y;
                //rank[x]++;
                rank[y]++;
            }
            return true;
        }
        return false;
    }
public:
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        sort(edges.begin(), edges.end(), [] (vector<int> &a, vector<int> &b) { return a[0] > b[0]; });
        vector<int> parenta(n+1),ranka(n+1,1);
        vector<int> parentb(n+1),rankb(n+1,1);
        for(int i=0;i<n;++i){
            parenta[i]=i;
            parentb[i]=i;
        }
        int mrga=1,mrgb=1,rmv=0;
        for(auto &e : edges){
            if(e[0]==3){
                bool tmpa=union_(e[1],e[2],parenta,ranka);
                bool tmpb=union_(e[1],e[2],parentb,rankb);
                if(tmpa==true) mrga++;
                if(tmpb==true) mrgb++;
                if(tmpa==false&&tmpb==false) rmv++;
            }
            else if(e[0]==1){
                bool tmpa=union_(e[1],e[2],parenta,ranka);
                if(tmpa==true) mrga++;
                else rmv++;
            }
            else{
                bool tmpb=union_(e[1],e[2],parentb,rankb);
                if(tmpb==true) mrgb++;
                else rmv++;
            }
        }
        if(mrga!=n || mrgb!=n) return -1;
        else return rmv;
    }
};