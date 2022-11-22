Find the Maximum Flow
========================
// Given a graph which represents a flow network with N vertices numbered 
// 1 to N and M edges.Find the maximum flow from vertex numbered 1 to vertex 
// numbered N.

// In a flow network,every edge has a flow capacity and the maximum flow of a 
// path can't exceed the flow-capacity of an edge in the path.

// Example 1:

// Input:
// N = 5, M =  4
// Edges[]= {{1,2,1},{3,2,2},{4,2,3},{2,5,5}}
// Output: 1 
// Explanation: 
// 1 - 2 - 3
//    / \
//   4   5 
// 1 unit can flow from 1 -> 2 - >5 

 

// Example 2:

// Input:
// N = 4, M = 4
// Edges[] = {{1,2,8},{1,3,10},{4,2,2},{3,4,3}}
// Output: 5 
// Explanation:
//   1 - 2 
//   |   |
//   3 - 4
// 3 unit can flow from 1 -> 3 -> 4
// 2 unit can flow from 1 -> 2 -> 4
// Total max flow from 1 to N = 3+2=5

// Your Task: 
// You don't need to read input or print anything. Your task is to complete the 
// function solve() which takes the N (the number of vertices) ,
// M (the number of Edges) and the array Edges[] (Where Edges[i] denoting an 
// undirected edge between Edges[i][0] and Edges[i][1] with a flow capacity of 
// Edges[i][2]), and returns the integer denoting the maximum flow from 1 to N.

// Expected Time Complexity: O(max_flow*M)
// Expected Auxiliary Space: O(N+M)

// Where max_flow is the maximum flow from 1 to N

// Constraints:
// 1 <= N,M,Edges[i][2] <= 1000
// 1 <= Edges[i][0],Edges[i][1] <= N

// solution -1
struct node
{
    int index;
    int cost;
    struct node *next;
};


class Solution
{
public:
    int maxval;
    struct node graph[5001];
    Solution()
    {
        maxval=1000000001;
    }
    void add(int u,int v,int w)
    {
    	int flag=0;
    	struct node *start = &graph[u];
    	while(start->next!=NULL)
    	{
    		if((start->next)->index == v)
    		{
    	    	(start->next)->cost+=w;
    			flag=1;
    			break;
    		}
    		start=start->next;
    	}
    	if(flag==0)
    	{
    	    struct node *temp=(struct node*)malloc(sizeof(struct node));
            temp->index=v;
            temp->cost=w;
            temp->next=NULL;
    		start->next=temp;
    	}
    	flag=0;
    
    	struct node *start2 = &graph[v];
        while(start2->next!=NULL)
        {
            if((start2->next)->index == u)
            {
                (start2->next)->cost+=w;
                flag=1;
                break;
            }
            start2=start2->next;
        }
        if(flag==0)
        {
            struct node *temp2=(struct node*)malloc(sizeof(struct node));
            temp2->index=u;
            temp2->cost=w;
            temp2->next=NULL;
            start2->next=temp2;
        }
    }
    
    int min(int val, int u,int v)
    {
    	int cost;
    	struct node *start2 = &graph[u];
        while(start2->next!=NULL)
        {
            if((start2->next)->index == v)
            {
                cost = (start2->next)->cost;
                break;
            }
            start2=start2->next;
        }
    	
    	if(cost<val)
    		return cost;
    	else
    		return val;
    }
    
    void insert(int u,int v,int flow)
    {
    	struct node *start = &graph[u];
        while(start->next!=NULL)
        {
            if((start->next)->index == v)
            {
                ((start->next)->cost)-=flow;
                break;
            }
            start=start->next;
        }
    
        struct node *start2 = &graph[v];
        while(start2->next!=NULL)
        {
            if((start2->next)->index == u)
            {
                ((start2->next)->cost)+=flow;
                break;
            }
            start2=start2->next;
        }
    }
    
    bool bfs(int s,int t,int parent[])
    {
    	bool visited[5001];
    	queue<int> q;
    	q.push(s);
    	for(int i=0;i<5001;i++)
    		visited[i]=false;
    	parent[s]=-1;
    	visited[s]=true;
    	while(!q.empty())
    	{
    		int u=q.front();
    		q.pop();
    		struct node *start2 = &graph[u];
    	    while(start2->next!=NULL)
            {
                if((start2->next)->cost>0 && visited[(start2->next)->index]==false)
                {
                    visited[(start2->next)->index]=true;
    		       	q.push((start2->next)->index);
    		        parent[(start2->next)->index]=u;
                }
                start2=start2->next;
            }
    	}
    	if(visited[t]==true)
    		return 1;
    	else return 0;
    }
    
    long long int fordfulkerson(int s, int t)
    {	
    	int parent[5001];
    	long long int max_flow=0;
    	int path_flow;
    	int u,v;
    
    	while(bfs(s,t,parent))
    	{
    		path_flow = maxval;
    		
    		for(v=t;v!=s;v=parent[v])
    		{
    			u=parent[v];
    			path_flow=min(path_flow,u,v);
    		}	
    
    		for(v=t;v!=s;v=parent[v])
    		{
    			u=parent[v];
    			insert(u,v,path_flow);
    		}
    		max_flow+=path_flow;	
    	}
    	return max_flow;
    }
    
    int findMaxFlow(int N,int M,vector<vector<int>> Edges)
    {
        int i,j,u,v,w;
        memset(graph,0,sizeof graph);
        for(j=0;j<=N;j++)
        {
        	graph[j].index=j;
        	graph[j].cost=0;
        	graph[j].next=NULL;
        }
        for(i=0;i<M;i++)
        {
            int u = Edges[i][0];
            int v = Edges[i][1];
            int w = Edges[i][2];
            if(u!=v)
            	add(u,v,w);	
        }
        return fordfulkerson(1,N);
    }
};

//solution -2 

class Solution
{
public:

    int bfs(int source, int sink,vector<vector<int>> &g, int N,vector<int> &parent){
            queue<pair<int,int>>q;
            vector<bool> vis(N,0);
            
            q.push({source,INT_MAX});
            vis[source]=1;
            
            while(!q.empty()){
                source=q.front().first;
                int cap=q.front().second;
                q.pop();
                
                for(int i=0;i<N;i++){
                    if(g[source][i] && !vis[i]){
                        parent[i]=source;
                        
                        if(i==sink)
                          return min(cap,g[source][i]);
                          
                        q.push({i,min(cap,g[source][i])});
                           vis[i]=1;
                    }
                }
            }
            return 0;
    }
    
    int ford_fulkerson(int source,int sink,vector<vector<int>> &g,int N){
        int flow=0;
        vector<int> parent(N,-1);
        
        int min_cap;
        while(min_cap=bfs(source,sink,g,N,parent)){
            flow+=min_cap;
            
            int u,v=sink;
            
            while(v!=source){
                  u=parent[v];
                  g[u][v]-=min_cap;
                  g[v][u]+=min_cap;
                  v=u;
            }
        }
        return flow; 

    }
    int findMaxFlow(int N,int M,vector<vector<int>> Edges)
    {
        // code hered
        vector<vector<int>> g(N+1,vector<int> (N+1,0));
        
        int i;
        for(int i=0;i<M;i++){
            g[Edges[i][0]][Edges[i][1]]+=Edges[i][2];
            g[Edges[i][1]][Edges[i][0]]+=Edges[i][2];
            
        }
        
        return ford_fulkerson(1,N,g,N+1);
    }
};

//ford-fulkerson

// C++ program for implementation of Ford Fulkerson
// algorithm
#include <iostream>
#include <limits.h>
#include <queue>
#include <string.h>
using namespace std;

// Number of vertices in given graph
#define V 6

/* Returns true if there is a path from source 's' to sink
't' in residual graph. Also fills parent[] to store the
path */
bool bfs(int rGraph[V][V], int s, int t, int parent[])
{
    // Create a visited array and mark all vertices as not
    // visited
    bool visited[V];
    memset(visited, 0, sizeof(visited));

    // Create a queue, enqueue source vertex and mark source
    // vertex as visited
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    // Standard BFS Loop
    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (visited[v] == false && rGraph[u][v] > 0) {
                // If we find a connection to the sink node,
                // then there is no point in BFS anymore We
                // just have to set its parent and can return
                // true
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    // We didn't reach sink in BFS starting from source, so
    // return false
    return false;
}

// Returns the maximum flow from s to t in the given graph
int fordFulkerson(int graph[V][V], int s, int t)
{
    int u, v;

    // Create a residual graph and fill the residual graph
    // with given capacities in the original graph as
    // residual capacities in residual graph
    int rGraph[V]
            [V]; // Residual graph where rGraph[i][j]
                // indicates residual capacity of edge
                // from i to j (if there is an edge. If
                // rGraph[i][j] is 0, then there is not)
    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
            rGraph[u][v] = graph[u][v];

    int parent[V]; // This array is filled by BFS and to
                // store path

    int max_flow = 0; // There is no flow initially

    // Augment the flow while there is path from source to
    // sink
    while (bfs(rGraph, s, t, parent)) {
        // Find minimum residual capacity of the edges along
        // the path filled by BFS. Or we can say find the
        // maximum flow through the path found.
        int path_flow = INT_MAX;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // update residual capacities of the edges and
        // reverse edges along the path
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        // Add path flow to overall flow
        max_flow += path_flow;
    }

    // Return the overall flow
    return max_flow;
}

// Driver program to test above functions
int main()
{
    // Let us create a graph shown in the above example
    int graph[V][V]
        = { { 0, 16, 13, 0, 0, 0 }, { 0, 0, 10, 12, 0, 0 },
            { 0, 4, 0, 0, 14, 0 }, { 0, 0, 9, 0, 0, 20 },
            { 0, 0, 0, 7, 0, 4 }, { 0, 0, 0, 0, 0, 0 } };

    cout << "The maximum possible flow is "
        << fordFulkerson(graph, 0, 5);

    return 0;
}
