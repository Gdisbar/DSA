Strongly Connected Components (Kosaraju''s Algo)
===================================================
// Given a Directed Graph with V vertices (Numbered from 0 to V-1) and E edges, 
// Find the number of strongly connected components in the graph.
 

// Example 1:

// Input:

// Output:
// 3
// Explanation:

// We can clearly see that there are 3 Strongly
// Connected Components in the Graph

// Example 2:

// Input:

// Output:
// 1
// Explanation:
// All of the nodes are connected to each other.
// So, there''s only one SCC.



class Solution{
public:
	//Function to find number of strongly connected components in the graph.
	void dfs(int node, stack<int> &st, vector<int> &vis, vector<int> adj[]) {
	    vis[node] = 1; 
	    for(auto it: adj[node]) {
	        if(!vis[it]) {
	            dfs(it, st, vis, adj); 
	        }
	    }
	    
	    st.push(node); //for toposort
    }
	void revDfs(int node, vector<int> &vis, vector<int> transpose[]) {
	   
	    vis[node] = 1; 
	    for(auto it: transpose[node]) {
	        if(!vis[it]) {
	            revDfs(it, vis, transpose); 
	        }
	    }
	}
    int kosaraju(int n, vector<int> adj[])
    {
        //code here
        stack<int> st;
        
		vector<int> vis(n, 0); 
		for(int i = 0;i<n;i++) {
		    if(!vis[i]) {
		        dfs(i, st, vis, adj); 
		    }
		}
		vector<int> transpose[n]; 
		
		for(int i = 0;i<n;i++) {
		    vis[i] = 0; // for revdfs
		    for(auto it: adj[i]) {
		        transpose[it].push_back(i); //transpose
		    }
		}
		int c=0;
		while(!st.empty()) {
		    int node = st.top();
		    st.pop(); 
		    if(!vis[node]) {
		         c++;
		        revDfs(node, vis, transpose); 
		       
		    }
		}
		return c;
    }
};

int main()
{
    
    int t;
    cin >> t;
    while(t--)
    {
    	int V, E;
    	cin >> V >> E;

    	vector<int> adj[V];

    	for(int i = 0; i < E; i++)
    	{
    		int u, v;
    		cin >> u >> v;
    		adj[u].push_back(v);
    	}

    	Solution obj;
    	cout << obj.kosaraju(V, adj) << "\n";
    }

    return 0;
}