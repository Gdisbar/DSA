Possible Path
=================
// Given an undirected graph with n vertices and connections between them. 
// Your task is to find whether you can come to same vertex X if you start from 
// X by traversing all the vertices atleast once and use all the paths exactly once.
 

// Example 1:

// Input: paths = {{0,1,1,1,1},{1,0,-1,1,-1},
// {1,-1,0,1,-1},{1,1,1,0,1},{1,-1,-1,1,0}}
// Output: 1
// Exaplanation: One can visit the vertices in
// the following way:
// 1->3->4->5->1->4->2->1
// Here all the vertices has been visited and all
// paths are used exactly once.


int isPossible(vector<vector<int>>paths){
	    int n = paths.size();
	    vector<int> adj[n+1];
	    
	    for(int i=0;i<n;i++){
	        for(int j=0;j<n;j++){
	            if(paths[i][j]==1){
	                adj[i].push_back(j);
	            }
	        }
	    }
	    int odd=0;
	    for(int i=0;i<n;i++){
	        if(adj[i].size()&1){
	            return 0;
	        }
	    }
	    return 1;
	    
	}



// Adam is standing at point (a,b) in an infinite 2D grid. He wants to know 
// if he can reach point (x,y) or not. The only operation he can do is to move to 
// point (a+b,b),(a,a+b),(a-b,b), or (a,b-a) from some point (a,b) . It is given 
// that he can move to any point on this 2D grid,i.e., the points having positive 
// or negative X(or Y) co-ordinates.Tell Adam whether he can reach (x,y) or not.



// Sample Input

// 3 ---> T
// 1 1 2 3 ---> a,b,x,y
// 2 1 2 3
// 3 3 1 1

// Sample Output

// YES
// YES
// NO

// Explanation

//     (1,1) -> (2,1) -> (2,3)

#define unsigned long long int ulli
int main()
{
    short int t;
    ulli a,b,x,y;
    scanf("%hi",&t);
    while(t--)
    {
        scanf("%llu%llu%llu%llu",&a,&b,&x,&y);
        if(gcd(a,b)==gcd(x,y))
        printf("YES\n");
        else
        printf("NO\n");
    }
    return 0;
}
ulli gcd(ulli n1,ulli n2)
{
    while(n1!=n2)
    {
        if(n1 > n2)
            n1 -= n2;
        else
            n2 -= n1;
    }
    return n1;
}