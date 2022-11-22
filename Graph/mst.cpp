MST - Minimum Spanning Tree # SPOJ
-------------------------------------
#mst

Find the minimum spanning tree of the graph.
Input

On the first line there will be two integers N - the number of nodes and M - the number of edges. (1 <= N <= 10000), (1 <= M <= 100000)
M lines follow with three integers i j k on each line representing an edge between node i and j with weight k. The IDs of the nodes are between 1 and n inclusive. The weight of each edge will be <= 1000000.
Output

Single number representing the total weight of the minimum spanning tree on this graph. There will be only one possible MST.
Example

Input:
4 5
1 2 10
2 3 15
1 3 5
4 2 2
4 3 40

Output:
17



#include<bits/stdc++.h>
#define debug(x) cout<<#x<<" = "<<x<<endl;
#define ll long long int
#define S(x) scanf("%d",&x);
using namespace std;
struct edges
{
	ll x, y, weight;
} graph[200000];
struct subset{
	ll parent;
	ll rank;
} subsets[20000];

ll find(ll i)
{
	if(subsets[i].parent != i)
		subsets[i].parent = find(subsets[i].parent);	
	return subsets[i].parent;
}

void unions(ll x, ll y)
{
	ll xroot = find(x);
	ll yroot = find(y);

	if(subsets[xroot].rank > subsets[yroot].rank) 
		subsets[yroot].parent = xroot;
	else if(subsets[xroot].rank < subsets[yroot].rank)
		subsets[xroot].parent = yroot;
	else
	{
		subsets[yroot].parent = xroot;
		subsets[xroot].rank++;
	}
}
bool comp(struct edges a, struct edges b)
{
	if(a.weight<b.weight)
		return true;
	return false;	
}
int N,M;
ll kruskal()
{
	sort(graph,graph+N,comp);
	for(int i=0;i<M;++i)
		subsets[i]={i,0};	
	ll ans=0;
	for(int i=0;i<N;++i)
		if(find(graph[i].x) != find(graph[i].y))
		{
			unions(graph[i].x,graph[i].y);
			ans+=graph[i].weight;
		}
	return ans;	
}
int main()
{
	cin>>M;
	cin>>N;
	for(int i=0;i<N;++i)
	{
		ll x,y;
		cin>>x>>y>>graph[i].weight;
		graph[i].x=x-1;
		graph[i].y=y-1;
	}
	cout<<kruskal();
}

