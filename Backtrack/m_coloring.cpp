m Coloring Problem
======================
// Given an undirected graph and a number m, determine if the graph can be coloured 
// with at most m colours such that no two adjacent vertices of the graph are 
// colored with the same color. Here coloring of a graph means the assignment 
// of colors to all vertices. 

// Input-Output format: 

// Input: 

// A 2D array graph[V][V] where V is the number of vertices in graph and 
// graph[V][V] is an adjacency matrix representation of the graph. A value 
// graph[i][j] is 1 if there is a direct edge from i to j, otherwise graph[i][j] 
// is 0.
// An integer m is the maximum number of colors that can be used.

// Output: 
// An array color[V] that should have numbers from 1 to m. color[i] should 
// represent the color assigned to the ith vertex. The code should also return 
// false if the graph cannot be colored with m colors.

// Example: 
                                // (3)-----(2)
                                // |      / |
                                // |    /   |
                                // |  /     |
                                // (0)-----(1)
// Input:  
// graph = {0, 1, 1, 1},
//         {1, 0, 1, 0},
//         {1, 1, 0, 1},
//         {1, 0, 1, 0}
// Output: 
// Solution Exists: 
// Following are the assigned colors
//  1  2  3  2
// Explanation: By coloring the vertices 
// with following colors, adjacent 
// vertices does not have same colors

// Input: 
// graph = {1, 1, 1, 1},
//         {1, 1, 1, 1},
//         {1, 1, 1, 1},
//         {1, 1, 1, 1}
// Output: Solution does not exist.
// Explanation: No solution exits.


// Number of vertices in the graph
#define V 4
 
int color[V]={0};
int m = 3
// check if the colored
// graph is safe or not
bool isSafe(bool graph[V][V])
{
    // check for every edge
    for (int i = 0; i < V; i++)
        for (int j = i + 1; j < V; j++)
            if (graph[i][j] && color[j] == color[i])
                return false;
    return true;
}
 

bool graphColoring(bool graph[V][V], int m, int i=0)
{
    // if current index reached end
    if (i == V) {
       
        // if coloring is safe
        if (isSafe(graph, color)) {
           
            // Print the solution
            printSolution(color);
            return true;
        }
        return false;
    }
 
    // Assign each color from 1 to m
    for (int j = 1; j <= m; j++) {
        color[i] = j;
 
        // Recur of the rest vertices
        if (graphColoring(graph, m, i + 1))
            return true;
 
        color[i] = 0;
    }
 
    return false;
}
 



// Solution Exists: Following are the assigned colors 
//  1  2  3  2 

// Complexity Analysis: 

//     Time Complexity: O(m^V). 
//     There are total O(m^V) combination of colors. 
//     So time complexity is O(m^V). The upperbound time complexity remains the 
//     same but the average time taken will be less.
//     Space Complexity: O(V). 
//     Recursive Stack of graphColoring(…) function will require O(V) space.

// BFS
class node{
public:
    int color = 1;
    set<int> edges;
};
 
int canPaint(vector<node>& nodes, int n, int m){
    vector<int> visited(n + 1, 0);
    // maxColors used till now are 1 as all nodes are painted color 1
    int maxColors = 1
    for (int v = 1; v <= n; v++){
        if (visited[v])
            continue;
        visited[v] = 1;
        queue<int> q;
        q.push(v);
        while (!q.empty()){
            int top = q.front();
            q.pop();
            for (auto it=nodes[top].edges.begin();it!=nodes[top].edges.end();it++){
                if (nodes[top].color == nodes[*it].color)
                    nodes[*it].color += 1;
                maxColors= max(maxColors, max(nodes[top].color,nodes[*it].color));
                if (maxColors > m)
                    return 0;
                if (!visited[*it]) {
                    visited[*it] = 1;
                    q.push(*it);
                }
            }
        }
    }
 
    return 1;
}
 
// vector<node> nodes(n + 1);
// if(graph[i][j]){
//   nodes[i].edges.insert(i);
//   nodes[j].edges.insert(j);
// }





from queue import Queue
 
class node:
    color = 1
    edges = set()
 
def canPaint(nodes, n, m):
    visited = [0 for _ in range(n+1)]
    // # maxColors used till now are 1 as all nodes are painted color 1
    maxColors = 1
    for _ in range(1, n + 1):
        if visited[_]:
            continue
        visited[_] = 1
        q = Queue()
        q.put(_)
        while not q.empty():
            top = q.get()
            for _ in nodes[top].edges:
                if nodes[top].color == nodes[_].color:
                    nodes[_].color += 1
                maxColors = max(maxColors, max([top].color, nodes[_].color))
                if maxColors > m:
                    print(maxColors)
                    return 0
                if not visited[_]:
                    visited[_] = 1
                    q.put(_)
    return 1
 

if __name__ == "__main__":
     
    n = 4
    graph = [ [ 0, 1, 1, 1 ],
              [ 1, 0, 1, 0 ],
              [ 1, 1, 0, 1 ],
              [ 1, 0, 1, 0 ] ]
               
    # Number of colors
    m = 3 
 
    # Create a vector of n+1
    # nodes of type "node"
    # The zeroth position is just
    # dummy (1 to n to be used)
    nodes = []
    for _ in range(n+1):
        nodes.append(node())
 
    # Add edges to each node as
    # per given input
    for _ in range(n):
        for __ in range(n):
            if graph[_][__]:
                 
                # Connect the undirected graph
                nodes[_].edges.add(_)
                nodes[__].edges.add(__)
 
    # Display final answer
    print(canPaint(nodes, n, m))


// Complexity Analysis:

//     Time Complexity: O(V + E).
//     Space Complexity: O(V). For Storing Visited List.