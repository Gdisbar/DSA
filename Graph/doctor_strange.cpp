Doctor Strange
=================
// The earth is protected by N sanctums, destroying any of it will lead to invasion 
// on earth.
// The sanctums are connected by M bridges.
// Now , you being on dormammu's side , want to find the number of sanctum 
// destroying which will disconnect the sanctums.

// Example 1:

// Input:
// N = 5, M = 5
// arr[] = {{1,2},{1,3},{3,2},{3,4},{5,4}}
// Output : 2
// Explanation:
// 1.Removing 1 will not make graph disconnected
// (2--3--4--5).
// 2.Removing 2 will also not make 
// graph disconnected(1--3--4--5).
// 3.Removing 3 makes graph disconnected 
// (1--2 and 4--5).
// 4.Removing 4 makes graph disconnected 
// (1--2--3--1 and 5).
// 5.Removing 5 also doesn't makes 
// graph disconnected(3--1--2--3--4).
// 6. Therefore,there are two such vertices,
// 3 and 4,so the answer is 2.

// Example 2:

// Input : 
// N = 2, M = 1 
// arr[] = {{1, 2}}
// Output : 0

// Your Task:

// This is a function problem. The input is already taken care of by the driver code. 
// You only need to complete the function doctorStrange() that takes a number of 
// nodes (N), a number of edges (M), a 2-D matrix that contains connection between 
// nodes (graph), and return the number of sanctums when destroyed will disconnect 
// other sanctums of Earth. 

// Articulation point// Tarjan's Algo 

class Solution{
 public:
    void dfs(int u,vector<int>&dis,vector<int>&low,vector<int>&parent,
			vector<int>adj[],unordered_set<int>&s)
   {
       static int t=0;
       low[u]=dis[u]=t;
       t++;
       int children=0;
       
       for(auto v:adj[u])
       {
           if(dis[v]==-1)
           {
               children++;
               parent[v]=u;
               dfs(v,dis,low,parent,adj,s);
               
               low[u]=min(low[u],low[v]);
               
               if(parent[u]==-1 and children>1)
               s.insert(u);
               
               if(parent[u]!=-1 and low[v]>=dis[u])
               s.insert(u);
           }
           else if(parent[u]!=v)
           low[u]=min(low[u],dis[v]);
       }
   }
   int doctorStrange(int n, int m, vector<vector<int>> & graph)
   {
        vector<int>adj[n+1];
        
        for(int i=0;i<m;i++)
        {
            adj[graph[i][0]].push_back(graph[i][1]);
            adj[graph[i][1]].push_back(graph[i][0]);
        }
        
        vector<int>dis(n+1,-1),low(n+1,-1),parent(n+1,-1);
        unordered_set<int>s;
        
        for(int i=1;i<=n;i++)
        {
            if(dis[i]==-1)
            dfs(i,dis,low,parent,adj,s);
        }
        
        
        
        return s.size();
   }
};


//Articulation point using dfs

class Solution{
 public:
    vector<int> I;// tin
	vector<vector<int>> G;
	int id = 0, C = 0;
	int dfs(int v, int p) {
	    int low = I[v] = ++id, flag = 0;
	    for (auto& u: G[v]) {
	        if (u == p) continue;
	        int l = I[u];
	        if (!l)
	            l = dfs(u, v),
	            flag = flag || l > I[v];
	        low = min(low, l);
	    }
	    if (p>=0 && flag) C++;
	    return low;
	}
	int doctorStrange(int N, int M, vector<vector<int>> & graph) {
	    G.resize(N); I.resize(N);
	    for (int i = 0; i < M; i++) {
	        auto& e = graph[i];
	        G[e[0]-1].push_back(e[1]-1),
	        G[e[1]-1].push_back(e[0]-1);
	    }
	    dfs(0, -1);
	    return C;
	}
};


//{ Driver Code Starts.

int main()
 {
    int t;
    cin>>t;
    while(t--)
    {
        int n,m,i;
        cin>>n>>m;
        vector<vector<int>> g(n+1);
        for(i=0;i<m;i++)
        {
            int a,b;cin>>a>>b;
            g[i].push_back(a);
            g[i].push_back(b);
        }
        Solution ob;
        int ans = ob.doctorStrange(n, m, g);
        cout<<ans<<endl;
    }
	return 0;
}