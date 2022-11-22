Mother Vertex
================
// Given a Directed Graph, find a Mother Vertex in the Graph (if present). 
// A Mother Vertex is a vertex through which we can reach all the other 
// vertices of the Graph.
 

// Example 1:

// Input: 

// Output: 0
// Explanation: According to the given edges, all 
// nodes can be reaced from nodes from 0, 1 and 2. 
// But, since 0 is minimum among 0,1 and 3, so 0 
// is the output.

// Example 2:

// Input: 

// Output: -1
// Explanation: According to the given edges, 
// no vertices are there from where we can 
// reach all vertices. So, output is -1.

// Expected Time Complexity: O(V + E)
// Expected Space Compelxity: O(V)

// A Naive approach : 
// A trivial approach will be to perform a DFS/BFS on all the vertices and find 
// whether we can reach all the vertices from that vertex. This approach takes 
// O(V(E+V)) time, which is very inefficient for large graphs.

// Can we do better? 
// We can find a mother vertex in O(V+E) time. The idea is based on Kosaraju''s 
// Strongly Connected Component Algorithm. In a graph of strongly connected 
// components, mother vertices are always vertices of source component in 
// component graph. 

// Basically mother vertex is last visited vertex as in DFS a parent vertex 
// can be visited once all it''s children have been visited




class Solution 
{

    public:
    //Function to find a Mother Vertex in the Graph.
    vector<bool> vis;
    void dfs(int node,vector<int>adj[]){
        vis[node]=true;
        for(const auto &x : adj[node]){
            if(!vis[x])
                dfs(x,adj);
        }
        
    }
	int findMotherVertex(int V, vector<int>adj[])
	{
	    // Code here
	    vis.resize(V,false);
	    int last_vis=0;
	    for(int i=0;i<V;++i){
	        if(!vis[i]){
	            dfs(i,adj);
	            last_vis=i; 
	        }
	    }
	    fill(vis.begin(),vis.end(),false);
	    dfs(last_vis,adj);
	    for(int i=0;i<V;++i)
	        if(!vis[i]) return -1;
	    return last_vis;
	}
};


// class Graph
// {
//     int V;    // No. of vertices
//     list<int> *adj;    // adjacency lists

//     // A recursive function to print DFS starting from v
//     void DFSUtil(int v, vector<bool> &visited);
// public:
//     Graph(int V);
//     void addEdge(int v, int w);
//     int findMother();
// };

// Graph::Graph(int V)
// {
//     this->V = V;
//     adj = new list<int>[V];
// }

// // A recursive function to print DFS starting from v
// void Graph::DFSUtil(int v, vector<bool> &visited)
// {
//     // Mark the current node as visited and print it
//     visited[v] = true;

//     // Recur for all the vertices adjacent to this vertex
//     list<int>::iterator i;
//     for (i = adj[v].begin(); i != adj[v].end(); ++i)
//         if (!visited[*i])
//             DFSUtil(*i, visited);
// }

// void Graph::addEdge(int v, int w)
// {
//     adj[v].push_back(w); // Add w to v’s list.
// }

// // Returns a mother vertex if exists. Otherwise returns -1
// int Graph::findMother()
// {
//     // visited[] is used for DFS. Initially all are
//     // initialized as not visited
//     vector <bool> visited(V, false);

//     // To store last finished vertex (or mother vertex)
//     int v = 0;

//     // Do a DFS traversal and find the last finished
//     // vertex  
//     for (int i = 0; i < V; i++)
//     {
//         if (visited[i] == false)
//         {
//             DFSUtil(i, visited);
//             v = i;
//         }
//     }

//     // If there exist mother vertex (or vertices) in given
//     // graph, then v must be one (or one of them)

//     // Now check if v is actually a mother vertex (or graph
//     // has a mother vertex).  We basically check if every vertex
//     // is reachable from v or not.

//     // Reset all values in visited[] as false and do 
//     // DFS beginning from v to check if all vertices are
//     // reachable from it or not.
//     fill(visited.begin(), visited.end(), false);
//     DFSUtil(v, visited); 
//     for (int i=0; i<V; i++)
//         if (visited[i] == false)
//             return -1;

//     return v;
// }

// // Driver program to test above functions
// int main()
// {
//     // Create a graph given in the above diagram
//     Graph g(7);
//     g.addEdge(0, 1);
//     g.addEdge(0, 2);
//     g.addEdge(1, 3);
//     g.addEdge(4, 1);
//     g.addEdge(6, 4);
//     g.addEdge(5, 6);
//     g.addEdge(5, 2);
//     g.addEdge(6, 0);

//     cout << "A mother vertex is " << g.findMother();

//     return 0;
// }
