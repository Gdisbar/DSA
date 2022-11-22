Johnson''s Algorithm
========================
// Johnson's Algorithm uses both Dijkstra''s Algorithm and Bellman-Ford Algorithm. 
// Johnson's Algorithm can find the all pair shortest path even in the case of 
// the negatively weighted graphs. It uses the Bellman-Ford algorithm in order to 
// eliminate negative edges by using the technique of reweighting the given graph 
// and detecting the negative cycles.

// This problem of finding the all pair shortest path can also be solved using 
// Floyd warshall Algorithm but the Time complexity of the Floyd warshall 
// algorithm is O(V^3), which is a polynomial-time algorithm, on the other hand, 
// the Time complexity of Johnson’s Algorithm is O(v^2log(V + ElogV) which is much 
// more efficient than the Floyd warshall Algorithm.

// The working of Johnson''s Algorithm can be easily understood by dividing the 
// algorithm into three major parts which are as follows

Adding a base vertex 
Reweighting the edges
Finding all-pairs shortest path


                        2
                    <----<---
                    |   0   |
               |--->a -----> b
               |  1 | \    | 0
            0  |    |  \   | 
               |    - 3 \  -
               |    |    \ |
               |--- d ---> c
                       5

// To calculate the shortest path between every pair of vertices the idea is 
// to add a base vertex (vertex SSS in figure below) that is directly connected 
// to every vertex. In other words, the cost to reach each vertex from the newly 
// added base vertex SSS is zero. Since at this stage our graph contains 
// negative weight edges so we apply the Bellman-Ford Algorithm.


D(S,a)=D(S,d)+D(d,a)
D(S,a)= 0 + (-3)
D(S,a)= -3

D(S,b)= D(S,a)+D(a,b) 
D(S,b) = -3 + (-5) 
D(S,b) = -8

D(S,c)= D(S,a) + D(a,b) + D(b,c) 
D(S,c)= -3 + (-5) + 6 
D(S,c)= -2

D(S,d)=0 (Direct edge between S and d).

// Reweighting of the edges is done because of the fact that if the 
// weight of all the edges in a given graph G are non negative, we can find the 
// shortest paths between all pairs of vertices just by running Dijkstra''s 
// Algorithm once from each vertex which is more efficient than applying the 
// Bellman-Ford algorithm.

// Now we calculate the new weights by using the above formula.

// W(u, v) = W(u, v) + D(S,u) - D(S,v)


W(a, b) = W(a, b) + D(S,a) - D(S,b)
W(a, b) = -5 + (-3) - (-8)
W(a, b) = 0

W(b, c) = W(b, c) + D(S,b) - D(S,c)
W(b, c) = 6  + (-8) - (-2)
W(b, c) = 0

W(d, c) = W(d, c) + D(S,d) - D(S,c)
W(d, c) = 3 + 0 - (-2)
W(a, b) = 5

// Now, after applying Dijkstra''s Algorithm on each vertex individually, we get the 
// shortest distance between all pairs, which is as follows.

// Consider a as source vertex, the shortest distance from a to all other vertices 
// are as follows.

// D(a, a) = 0

D(a, b) = W(a, b)
D(a, b) = 0
D(a, c) = W(a, b) + W(b, c)
D(a, c) = 0 + 0 = 0
D(a, d) = W(a, d)
D(a, d) = 1
D(b, a)= W(b, a)
D(b, a)= 2



D(b, b)= 0
D(b, c) = W(b, c)
D(b, c) = 0
D(b, d)=W(b,a) +W(a, d)
D(b, d) = 2 + 1 = 3


D(c, a) = W(c,a)
D(c, a) = 3
D(c, b) = W(c, a) +W(a, b)
D(c, b) = 3 + 0
D(c, b) = 3
D(c, c) = 0
D(c, d) = W(c, a) + W(a,d)
D(c, d) =  3 + 1
D(c, d) = 4



D(d, a) = W(d, a)
D(d, a) = 0
D(d, b) = W(d, a) + W(a, b)
D(d, b) = 0 + 0 = 0
D(d, c) = W(d, a) + W(a, b) + W(b, c)
D(d, c) = 0 + 0 + 0 = 0
D(d, d) = 0

Distance    
    a   b   c   d
a   0   0   0   1
b   2   0   0   3
c   3   3   0   4
d   0   0   0   0


// Johnson''s_Algorithm(G):
    
//     1. Create a graph G` from the given graph G
//        such that G`.V = G.V + {s}
       
//        where s is a newly added base vertex 
//        G`.E = G.E + ((s, u) for u in G.V), 
//        and weight(s, u) = 0 for u in G.V
    
//     2. if Bellman-Ford(s) == False:
//             return "The input graph contains a negative weight cycle"
//         otherwise:
       
//             for each vertex v in G`.V:
//                 h(v) = distance(s, v) 
//                 which is computed by using 
//                 Bellman-Ford Algorithm
            
//             for every edge (u, v) in G`.E:
//                 weight`(u, v) = weight(u, v) + h(u) - h(v)
    
//     3.  Dist = new matrix that is used to store the shortest path 
//         between all pair of vertices in the graph G and 
//         initilize the whole matrix to infinity
        
//         for vertex u in G.V:
//             run Dijkstra(G, weight`, u) to compute distance`(u, v) for all v in G.V
        
//         for each vertex v in G.V:
//             D_(u, v) = distance`(u, v) + h(v) - h(u)
        
//         return Dist


# Implementation of Johnson''s algorithm in Python3

# Import function to initialize the dictionary
from collections import defaultdict
INT_MAX = float('Inf')

# Function that returns the vertex 
# with minimum distance 
# from the source
def Min_Distance(dist, visit):

    (minimum, Minimum_Vertex) = (INT_MAX, 0)
    for vertex in range(len(dist)):
        if minimum > dist[vertex] and visit[vertex] == False:
            (minimum, minVertex) = (dist[vertex], vertex)

    return Minimum_Vertex


# Dijkstra Algorithm for Modified
# Graph (After removing the negative weights)
def Dijkstra_Algorithm(graph, Altered_Graph, source):

    # Number of vertices in the graph
    tot_vertices = len(graph)

    # Dictionary to check if given vertex is
    # already included in the shortest path tree
    sptSet = defaultdict(lambda : False)

    # Shortest distance of all vertices from the source
    dist = [INT_MAX] * tot_vertices

    dist[source] = 0

    for count in range(tot_vertices):

        # The current vertex which is at min Distance
        # from the source and not yet included in the
        # shortest path tree
        curVertex = Min_Distance(dist, sptSet)
        sptSet[curVertex] = True

        for vertex in range(tot_vertices):
            if ((sptSet[vertex] == False) and
                (dist[vertex] > (dist[curVertex] +
                Altered_Graph[curVertex][vertex])) and
                (graph[curVertex][vertex] != 0)):
                                 
                                                    dist[vertex] = (dist[curVertex] +Altered_Graph[curVertex][vertex])

    # Print the Shortest distance from the source
    for vertex in range(tot_vertices):
        print ('Vertex ' + str(vertex) + ': ' + str(dist[vertex]))

# Function to calculate shortest distances from source
# to all other vertices using Bellman-Ford algorithm
def BellmanFord_Algorithm(edges, graph, tot_vertices):

    # Add a source s and calculate its min
    # distance from every other node
    dist = [INT_MAX] * (tot_vertices + 1)
    dist[tot_vertices] = 0

    for i in range(tot_vertices):
        edges.append([tot_vertices, i, 0])

    for i in range(tot_vertices):
        for (source, destn, weight) in edges:
            if((dist[source] != INT_MAX) and
                    (dist[source] + weight < dist[destn])):
                dist[destn] = dist[source] + weight

    # Don''t send the value for the source added
    return dist[0:tot_vertices]

# Function to implement Johnson Algorithm
def JohnsonAlgorithm(graph):

    edges = []

    # Create a list of edges for Bellman-Ford Algorithm
    for i in range(len(graph)):
        for j in range(len(graph[i])):

            if graph[i][j] != 0:
                edges.append([i, j, graph[i][j]])

    # Weights used to modify the original weights
    Alter_weigts = BellmanFord_Algorithm(edges, graph, len(graph))

    Altered_Graph = [[0 for p in range(len(graph))] for q in
                    range(len(graph))]

    # Modify the weights to get rid of negative weights
    for i in range(len(graph)):
        for j in range(len(graph[i])):

            if graph[i][j] != 0:
                Altered_Graph[i][j] = (graph[i][j] +
                        Alter_weigts[i] - Alter_weigts[j]);

    print ('Modified Graph: ' + str(Altered_Graph))

    # Run Dijkstra for every vertex as source one by one
    for source in range(len(graph)):
        print ('\nShortest Distance with vertex ' +
                        str(source) + ' as the source:\n')
        Dijkstra_Algorithm(graph, Altered_Graph, source)

# Driver Code
graph = [[0, -5, 2, 3],
        [0, 0, 4, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]]

JohnsonAlgorithm(graph)


