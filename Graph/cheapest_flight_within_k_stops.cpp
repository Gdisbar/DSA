787. Cheapest Flights Within K Stops
======================================
There are n cities connected by some number of flights. You are given an array 
flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight 
from city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price 
from src to dst with at most k stops. If there is no such route, return -1.

 

Example 1:

Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 
src = 0, dst = 3, k = 1
Output: 700
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 3 is marked in red and has 
cost 100 + 600 = 700.
Note that the path through cities [0,1,2,3] is cheaper but is invalid because 
it uses 2 stops.

Example 2:

Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 2 is marked in red and has 
cost 100 + 100 = 200.

Example 3:

Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0
Output: 500
Explanation:
The graph is shown above.
The optimal path with no stops from city 0 to 2 is marked in red and has cost 500.

// TLE
#define pb push_back
class Solution {
    void solve(vector<vector<int>>& adj,vector<vector<int>>& cost,int src,
    	int dst,int k,int& fare,int totCost,vector<bool>& visited)
    {
        if(k<-1)
            return;
        if(src==dst)
        {
            fare = min(fare,totCost);
            return;
        }
    
        visited[src] = true;
        for(int i=0;i<adj[src].size();++i)
        {
            if(!visited[adj[src][i]] && (totCost+cost[src][adj[src][i]] <= fare))
                solve(adj,cost,adj[src][i],dst,k-1,fare,totCost+cost[src][adj[src][i]],visited);
        }
        
        visited[src] = false;
    }
public:
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) 
    {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);
        cout.tie(NULL);
        
        vector<vector<int>> adj(n);
        vector<vector<int>> cost(n+1,vector<int>(n+1));
        
        for(int i=0;i<flights.size();++i)
        {    
            adj[flights[i][0]].pb(flights[i][1]);
            cost[flights[i][0]][flights[i][1]] = flights[i][2];
        }
        
        vector<bool> visited(n+1,false);    //To break cycles
        int fare = INT_MAX;
        solve(adj,cost,src,dst,K,fare,0,visited);
        
        if(fare==INT_MAX)
            return -1;
        return fare;
    }
};

//dfs

int findCheapestPrice(int n, vector<vector<int>>& flights, int s, int d, int K) {
       unordered_map<int, vector<pair<int,int>>> g;
       for (const auto& e : flights)
            g[e[0]].emplace_back(e[1], e[2]);        
        int ans = INT_MAX;
        vector<int> visited(n,0);
        dfs(s, d, K + 1, 0, visited, ans, g);
        return ans == INT_MAX ? - 1 : ans;
    }
    
    void dfs(int s, int d, int k, int cost, vector<int>& visited, int& ans, unordered_map<int, vector<pair<int,int>>>& g ) {
        if (s == d) { ans = cost; return; }
        if (k == 0) return; 
        visited[s]=1;
        for (const auto& x : g[s]) {
          if (visited[x.first]==0){
              if (cost + x.second > ans) continue; // IMPORTANT!!! prunning 
     
              dfs(x.first, d, k - 1, cost + x.second, visited, ans, g); 
             
          }
        }
         visited[s] = 0;
  }


// dijkstra

typedef tuple<int,int,int> ti;
class Solution {
public:
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        vector<vector<pair<int,int>>>vp(n);
        for(const auto&f:flights)   vp[f[0]].emplace_back(f[1],f[2]);
        priority_queue<ti,vector<ti>,greater<ti>>pq;
        pq.emplace(0,src,K+1);
        while(!pq.empty()){
            auto [cost,u,stops] = pq.top();
            pq.pop();
            if(u==dst)  return cost;
            if(!stops)  continue;
            for(auto to:vp[u]){
                auto [v,w] = to;
                pq.emplace(cost+w,v,stops-1);
            }
        }
        return -1;
    }
};

// bellmanford

int findCheapestPrice(int n, vector<vector<int>>& a, int src, int sink, int k) {
        
        vector<int> c(n, 1e8);
        c[src] = 0;
        
        for(int z=0; z<=k; z++){
            vector<int> C(c);
            for(auto e: a)
                C[e[1]] = min(C[e[1]], c[e[0]] + e[2]);
            c = C;
        }
        return c[sink] == 1e8 ? -1 : c[sink];
    }

//dp
 int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
    vector<vector<int>> dp(K + 2, vector<int> (n, INT_MAX));
    for(int i = 0; i <= K + 1; i++) dp[i][src] = 0;
    for(int i = 1; i <= K + 1; i++){
      for(auto& f:flights){
        if(dp[i-1][f[0]] != INT_MAX)
          dp[i][f[1]] = min(dp[i][f[1]], dp[i-1][f[0]] + f[2]);
      }
    }
    return dp[K+1][dst] == INT_MAX? -1: dp[K+1][dst];
  } 

    
 def findCheapestPrice(self, n, flights, src, dst, k):
        f = collections.defaultdict(dict)
        for a, b, p in flights:
            f[a][b] = p
        heap = [(0, src, k + 1)]
        while heap:
            p, i, k = heapq.heappop(heap)
            if i == dst:
                return p
            if k > 0:
                for j in f[i]:
                    heapq.heappush(heap, (p + f[i][j], j, k - 1))
        return -1


