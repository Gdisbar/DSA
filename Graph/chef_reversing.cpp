Chef and Reversing | Problem Code:REVERSE
==========================================
// Sometimes mysteries happen. Chef found a directed graph with N vertices 
// and M edges in his kitchen!

// The evening was boring and chef has nothing else to do, so to entertain himself, 
// Chef thought about a question What is the minimum number of edges he needs to 
// reverse in order to have at least one path from vertex 1 to vertex N, 
// where the vertices are numbered from 1 to N.


// 			  1 ---> 2 <--- 6 <--- 5 <--- 7 
// 			         ^                    |
// 			         |                    |
// 			         3 ---> 4 <------------

// We can consider two paths from 1 to 7:

//     1-2-3-4-7
//     1-2-6-5-7

// In the first one we need to revert edges (3-2), (7-4). 
// In the second one - (6-2), (5-6), (7-5). So the answer is min(2, 3) = 2.



// Add reverse edge of each original edge in the graph. Give reverse edge a 
// weight=1 and all original edges a weight of 0. Now, the length of the 
// shortest path will give us the answer. --> is it 0-1 BFS ?

// How?

// If shortest path is p: it means we used k reverse edges in the shortest path. 
// So, it will give us the answer.
// The shortest path algorithm will always try to use as less reverse paths 
// possible because they have higher weight than original edges.

// Dijkstra : O(|E| log |V|) with adjacency lists and priority queue.
// Also, since there are only 0 and 1 weight edges, 
// we can also do this by BFS: maintain a deque instead of a queue and add a 
// vertex to the front of the deque if 0 edge is used and to the back of the 
// deque otherwise.




int main(){
  int n,m;cin>>n>>m;
  vector<vector<vector<int>>> adj(n+1);
  while(m--){
    int u,v;cin>>u>>v;
    adj[u].push_back({v,0});
    adj[v].push_back({u,1});
  }
  //dijkstra''s 
  priority_queue<pair<int,int>> q;
  vector<int> dist(n+1,INT_MAX);
  q.push({0,1});
  dist[1]=0;
  while(!q.empty()){
      int u = q.top().second;
      q.pop();
      for(auto &val : adj[u]){
        int v =val[0];
        int w=val[1];
        if(dist[v]>dist[u]+w){
          dist[v]=dist[u]+w;
          q.push({-w,v}); //convert into min heap
        }
      }
  }
  if(dist[n]>=INT_MAX) cout<<-1;
  else cout<<dist[n];
  return 0;
}