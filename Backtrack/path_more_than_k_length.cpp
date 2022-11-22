Find if there is a path of more than k length from a source
===============================================================
// Given a graph, a source vertex in the graph and a number k, find if there is 
// a simple path (without any cycle) starting from given source and ending at any 
// other vertex such that the distance from source to that vertex is atleast ‘k’ 
// length.

// Example:
                   //          8    7     
                   //   ----1----2-----3--|
                   // 4 |   |  2 | \      | 9
                   //   0   11  /8   \ 4  4
                   // 8 |   | /7 | 6   \  |10
                   //   ----7----6---5-----
                   //         1    2

// Input  : Source s = 0, k = 58
// Output : True
// There exists a simple path 0 -> 7 -> 1
// -> 2 -> 8 -> 6 -> 5 -> 3 -> 4
// Which has a total distance of 60 km which
// is more than 58.

// Input  : Source s = 0, k = 62
// Output : False

// In the above graph, the longest simple
// path has distance 61 (0 -> 7 -> 1-> 2
//  -> 3 -> 4 -> 5-> 6 -> 8, so output 
// should be false for any input greater 
// than 61.




// iPair ==>  Integer Pair
typedef pair<int, int> iPair;
 
// This class represents a dipathted graph using
// adjacency list representation
class Graph
{
    int V;    // No. of vertices
 
    // In a weighted graph, we need to store vertex
    // and weight pair for every edge
    list< pair<int, int> > *adj;
    bool pathMoreThanKUtil(int src, int k, vector<bool> &path);
 
public:
    Graph(int V);  // Constructor
 
    // function to add an edge to graph
    void addEdge(int u, int v, int w);
    bool pathMoreThanK(int src, int k);
};
 
// Returns true if graph has path more than k length
bool Graph::pathMoreThanK(int src, int k)
{
    // Create a path array with nothing included
    // in path
    vector<bool> path(V, false);
 
    // Add source vertex to path
    path[src] = 1;
 
    return pathMoreThanKUtil(src, k, path);
}
 
// Prints shortest paths from src to all other vertices
bool Graph::pathMoreThanKUtil(int src, int k, vector<bool> &path)
{
    // If k is 0 or negative, return true;
    if (k <= 0)
        return true;
 
    // Get all adjacent vertices of source vertex src and
    // recursively explore all paths from src.
    list<iPair>::iterator i;
    for (i = adj[src].begin(); i != adj[src].end(); ++i)
    {
        // Get adjacent vertex and weight of edge
        int v = (*i).first;
        int w = (*i).second;
 
        // If vertex v is already there in path, then
        // there is a cycle (we ignore this edge)
        if (path[v] == true)
            continue;
 
        // If weight of is more than k, return true
        if (w >= k)
            return true;
 
        // Else add this vertex to path
        path[v] = true;
 
        // If this adjacent can provide a path longer
        // than k, return true.
        if (pathMoreThanKUtil(v, k-w, path))
            return true;
 
        // Backtrack
        path[v] = false;
    }
 
    // If no adjacent could produce longer path, return
    // false
    return false;
}
 
// Allocates memory for adjacency list
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<iPair> [V];
}
 
// Utility function to an edge (u, v) of weight w
void Graph::addEdge(int u, int v, int w)
{
    adj[u].push_back(make_pair(v, w));
    adj[v].push_back(make_pair(u, w));
}
 
// Driver program to test methods of graph class
int main()
{
    // create the graph given in above figure
    int V = 9;
    Graph g(V);
 
    //  making above shown graph
    g.addEdge(0, 1, 4);
    g.addEdge(0, 7, 8);
    g.addEdge(1, 2, 8);
    g.addEdge(1, 7, 11);
    g.addEdge(2, 3, 7);
    g.addEdge(2, 8, 2);
    g.addEdge(2, 5, 4);
    g.addEdge(3, 4, 9);
    g.addEdge(3, 5, 14);
    g.addEdge(4, 5, 10);
    g.addEdge(5, 6, 2);
    g.addEdge(6, 7, 1);
    g.addEdge(6, 8, 6);
    g.addEdge(7, 8, 7);
 
    int src = 0;
    int k = 62;
    g.pathMoreThanK(src, k)? cout << "Yes\n" :cout << "No\n";
 
    k = 60;
    g.pathMoreThanK(src, k)? cout << "Yes\n" :cout << "No\n";
 
    return 0;
}

Output: 
 

No
Yes


//Time Complexity: O(n!) 


class Graph:
    
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
   
    def pathMoreThanK(self,src, k): 
        path = [False]*self.V
        path[src] = 1
        return self.pathMoreThanKUtil(src, k, path)
       
    def pathMoreThanKUtil(self,src, k, path):
        if (k <= 0):
            return True

        i = 0
        while i != len(self.adj[src]):
            v = self.adj[src][i][0]
            w = self.adj[src][i][1]
            i += 1
            if (path[v] == True):
                continue
            if (w >= k):
                return True
            path[v] = True
            if (self.pathMoreThanKUtil(v, k-w, path)):
                return True
            path[v] = False

        return False

    def addEdge(self,u, v, w):
        self.adj[u].append([v, w])
        self.adj[v].append([u, w])
   

if __name__ == '__main__':
  
    # create the graph given in above figure
    V = 9
    g = Graph(V)
   
    #  making above shown graph
    g.addEdge(0, 1, 4)
    g.addEdge(0, 7, 8)
    g.addEdge(1, 2, 8)
    g.addEdge(1, 7, 11)
    g.addEdge(2, 3, 7)
    g.addEdge(2, 8, 2)
    g.addEdge(2, 5, 4)
    g.addEdge(3, 4, 9)
    g.addEdge(3, 5, 14)
    g.addEdge(4, 5, 10)
    g.addEdge(5, 6, 2)
    g.addEdge(6, 7, 1)
    g.addEdge(6, 8, 6)
    g.addEdge(7, 8, 7)
   
    src = 0
    k = 62
    if g.pathMoreThanK(src, k):
        print("Yes")
    else:
        print("No")
   
    k = 60
    if g.pathMoreThanK(src, k):
        print("Yes")
    else:
        print("No")