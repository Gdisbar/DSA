Kruskal’s Minimum Spanning Tree Algorithm
=============================================
// Kruskal’s algorithm to find the minimum cost spanning tree uses the 
// greedy approach. The Greedy Choice is to pick the smallest weight edge 
// that does not cause a cycle in the MST constructed so far. Let us understand 
// it with an example:

// Below is the illustration of the above approach:

//     Input Graph:
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     The graph contains 9 vertices and 14 edges. So, the minimum spanning 
//     tree formed will be having (9 – 1) = 8 edges. 

//     After sorting:
//     Weight   Src    Dest
//     1         7      6
//     2         8      2
//     2         6      5
//     4         0      1
//     4         2      5
//     6         8      6
//     7         2      3
//     7         7      8
//     8         0      7
//     8         1      2
//     9         3      4
//     10        5      4
//     11        1      7
//     14        3      5

//     Now pick all edges one by one from the sorted list of edges 
//     Step 1: Pick edge 7-6: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 2:  Pick edge 8-2: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 3: Pick edge 6-5: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 4: Pick edge 0-1: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 5: Pick edge 2-5: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 6: Pick edge 8-6: Since including this edge results in the cycle, 
//     discard it.
//     Step 7: Pick edge 2-3: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 8: Pick edge 7-8: Since including this edge results in the cycle, 
//     discard it.
//     Step 9: Pick edge 0-7: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Step 10: Pick edge 1-2: Since including this edge results in the cycle, 
//     discard it.
//     Step 11: Pick edge 3-4: No cycle is formed, include it. 
     

//     Kruskal’s Minimum Spanning Tree Algorithm

//     Note: Since the number of edges included in the MST equals to (V – 1), 
//     so the algorithm stops here.


    // DSU data structure
// path compression + rank by union
 
class DSU {
    int* parent;
    int* rank;
 
public:
    DSU(int n)
    {
        parent = new int[n];
        rank = new int[n];
 
        for (int i = 0; i < n; i++) {
            parent[i] = -1;
            rank[i] = 1;
        }
    }
 
    // Find function
    int find(int i)
    {
        if (parent[i] == -1)
            return i;
 
        return parent[i] = find(parent[i]);
    }
   
    // Union function
    void unite(int x, int y)
    {
        int s1 = find(x);
        int s2 = find(y);
 
        if (s1 != s2) {
            if (rank[s1] < rank[s2]) {
                parent[s1] = s2;
                rank[s2] += rank[s1];
            }
            else {
                parent[s2] = s1;
                rank[s1] += rank[s2];
            }
        }
    }
};
 
class Graph {
    vector<vector<int> > edgelist;
    int V;
 
public:
    Graph(int V) { this->V = V; }
 
    void addEdge(int x, int y, int w)
    {
        edgelist.push_back({ w, x, y });
    }
 
    void kruskals_mst()
    {
        // 1. Sort all edges
        sort(edgelist.begin(), edgelist.end());
 
        // Initialize the DSU
        DSU s(V);
        int ans = 0;
        cout << "Following are the edges in the "
                "constructed MST"
             << endl;
        for (auto edge : edgelist) {
            int w = edge[0];
            int x = edge[1];
            int y = edge[2];
 
            // Take this edge in MST if it does
              // not forms a cycle
            if (s.find(x) != s.find(y)) {
                s.unite(x, y);
                ans += w;
                cout << x << " -- " << y << " == " << w
                     << endl;
            }
        }
       
        cout << "Minimum Cost Spanning Tree: " << ans;
    }
};
 
// Driver's code
int main()
{
    /* Let us create following weighted graph
                   10
              0--------1
              |  \     |
             6|   5\   |15
              |      \ |
              2--------3
                  4       */
    Graph g(4);
    g.addEdge(0, 1, 10);
    g.addEdge(1, 3, 15);
    g.addEdge(2, 3, 4);
    g.addEdge(2, 0, 6);
    g.addEdge(0, 3, 5);
 
      // Function call
    g.kruskals_mst();
    return 0;
}

// Output

// Following are the edges in the constructed MST
// 2 -- 3 == 4
// 0 -- 3 == 5
// 0 -- 1 == 10
// Minimum Cost Spanning Tree: 19

# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph
 
from collections import defaultdict
 
# Class to represent a graph
 
 
class Graph:
 
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph
 
    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
 
    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
 
        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
 
    # The main function to construct MST using Kruskal's
        # algorithm
    def KruskalMST(self):
 
        result = []  # This will store the resultant MST
 
        # An index variable, used for sorted edges
        i = 0
 
        # An index variable, used for result[]
        e = 0
 
        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])
 
        parent = []
        rank = []
 
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
 
        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:
 
            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
 
            # If including this edge doesn't
            # cause cycle, then include it in result
            # and increment the index of result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge
 
        minimumCost = 0
        print("Edges in the constructed MST")
        for u, v, weight in result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree", minimumCost)
 
 
# Driver's code
if __name__ == '__main__':
    g = Graph(4)
    g.addEdge(0, 1, 10)
    g.addEdge(0, 2, 6)
    g.addEdge(0, 3, 5)
    g.addEdge(1, 3, 15)
    g.addEdge(2, 3, 4)
 
    # Function call
    g.KruskalMST()
 
# This code is contributed by Neelam Yadav
Output

Following are the edges in the constructed MST
2 -- 3 == 4
0 -- 3 == 5
0 -- 1 == 10
Minimum Cost Spanning Tree: 19