Bellman–Ford Algorithm
=======================
// Given a graph and a source vertex src in the graph, find the shortest paths 
// from src to all vertices in the given graph. The graph may contain negative 
// weight edges.

// Dijkstra’s algorithm is a Greedy algorithm and the time complexity is 
// O((V+E)LogV) (with the use of the Fibonacci heap). Dijkstra doesn’t work for 
// Graphs with negative weights, Bellman-Ford works for such graphs. Bellman-Ford 
// is also simpler than Dijkstra and suites well for distributed systems. 
// But time complexity of Bellman-Ford is O(V * E) 


void BellmanFord(int graph[][3], int V, int E,int src)
{
    // Initialize distance of all vertices as infinite.
    int dis[V];
    for (int i = 0; i < V; i++)
        dis[i] = INT_MAX;
 
    // initialize distance of source as 0
    dis[src] = 0;
 
    // Relax all edges |V| - 1 times. A simple
    // shortest path from src to any other
    // vertex can have at-most |V| - 1 edges
    //dis[u]+w < dis[v]
    for (int i = 0; i < V - 1; i++) {
        for (int j = 0; j < E; j++) {
            if (dis[graph[j][0]] != INT_MAX && 
            	dis[graph[j][0]] + graph[j][2] <dis[graph[j][1]])
                dis[graph[j][1]] = dis[graph[j][0]] + graph[j][2];
        }
    }
 
    // check for negative-weight cycles.
    // The above step guarantees shortest
    // distances if graph doesn't contain
    // negative weight cycle.  If we get a
    // shorter path, then there is a cycle.
    for (int i = 0; i < E; i++) {
        int x = graph[i][0];
        int y = graph[i][1];
        int weight = graph[i][2];
        if (dis[x] != INT_MAX && dis[x] + weight < dis[y])
            cout << "Graph contains negative weight cycle"<< endl;
    }
 
    cout << "Vertex Distance from Source" << endl;
    for (int i = 0; i < V; i++)
        cout << i << "\t\t" << dis[i] << endl;
}
 
Vertex Distance from Source
0        0
1        -1
2        2
3        -2
4        1


# Python3 program for Bellman-Ford''s
# single source shortest path algorithm.
from sys import maxsize
 
# The main function that finds shortest
# distances from src to all other vertices
# using Bellman-Ford algorithm. The function
# also detects negative weight cycle
# The row graph[i] represents i-th edge with
# three values u, v and w.
def BellmanFord(graph, V, E, src):
 
    # Initialize distance of all vertices as infinite.
    dis = [maxsize] * V
 
    # initialize distance of source as 0
    dis[src] = 0
 
    # Relax all edges |V| - 1 times. A simple
    # shortest path from src to any other
    # vertex can have at-most |V| - 1 edges
    for i in range(V - 1):
        for j in range(E):
            if dis[graph[j][0]] + graph[j][2] < dis[graph[j][1]]:
                dis[graph[j][1]] = dis[graph[j][0]] + graph[j][2]
 
    # check for negative-weight cycles.
    # The above step guarantees shortest
    # distances if graph doesn''t contain
    # negative weight cycle. If we get a
    # shorter path, then there is a cycle.
    for i in range(E):
        x = graph[i][0]
        y = graph[i][1]
        weight = graph[i][2]
        if dis[x] != maxsize and dis[x] + weight < dis[y]:
            print("Graph contains negative weight cycle")
 
    print("Vertex Distance from Source")
    for i in range(V):
        print("%d\t\t%d" % (i, dis[i]))
 
# Driver Code
if __name__ == "__main__":
    V = 5 # Number of vertices in graph
    E = 8 # Number of edges in graph
 
    # Every edge has three values (u, v, w) where
    # the edge is from vertex u to v. And weight
    # of the edge is w.
    graph = [[0, 1, -1], [0, 2, 4], [1, 2, 3],
             [1, 3, 2], [1, 4, 2], [3, 2, 5],
             [3, 1, 1], [4, 3, -3]]
    BellmanFord(graph, V, E, 0)
 