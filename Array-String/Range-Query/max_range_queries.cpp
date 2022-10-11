Max Range Queries - Problem Code:MAXREMOV
=============================================
// You have C=100,000 cakes, numbered 1 through C. Each cake has an integer height; 
// initially, the height of each cake is 0.There are N operations. In each operation, 
// you are given two integers L and R, and you should increase by 1 the height of 
// each of the cakes L,L+1,…,R. One of these N operations should be removed and the 
// remaining N−1operations are then performed.
// Chef wants to remove one operation in such a way that after the remaining 
// N−1 operations are performed, the number of cakes with height exactly K is maximum 
// possible. Since Chef is a bit busy these days, he has asked for your help. 
// You need to find the maximum number of cakes with height exactly KKK that can be 
// achieved by removing one operation.

// Input

// The first line of the input contains a single integer T denoting the number of test 
// cases. The description of T test cases follows.
// The first line of each test case contains two space-separated integers N and K.
// Each of the next NNN lines contains two space-separated integers L and R describing 
// one operation.

// Output

// For each test case, print a single line containing one integer — the maximum 
// possible number of cakes with height K.



// Sample 1:
// Input
// Output

// 1
// 3 2
// 2 6
// 4 9
// 1 4

// 3

// Explanation:

// Example case 1: Let''s look at what happens after an operation is removed.

// Removing operation 1: The heights of cakes 4 through 9 increase by 1. 
// Then, the heights of cakes 1 through 4 increase by 1. The resulting 
// sequence of heights is [1,1,1,2,1,1,1,1,1]
// (for cakes 1 through 9; the other cakes have heights 0). 
// The number of cakes with height 2 is 1.
// Removing operation 2: The resulting sequence of heights of cakes 1 through 9 
// is [1,2,2,2,1,1,0,0,0]. 
// The number of cakes with height 2 is 3.
// Removing operation 3: The resulting sequence of heights of cakes 1 through 9 
// is [0,1,1,2,2,2,1,1,1]. The number of cakes with height 2 is 3.

// The maximum number of cakes with height 2 is 3.


// EXPLANATION:

// We will discuss a linear solution to this problem. Let’s first assume that all 
// operations are performed. How can we calculate cakes’ heights after all 
// operations are finished?

// Let’s read all the operations first. Then, let’s iterate through cakes from the 
// first one till the last one. Suppose that we are processing currently the i-th 
// cake. If there is some operation which has L=i that means that we must increase the 
// heights of all cakes while we are moving forward until R+1=i. When R+1=i that 
// mentioned operation won’t affect any more cakes.

// So we should keep an array change[1...C].

// change[i]= # of queries with L=i minus the number of queries with R=i−1.

// (R=i−1 because the R-th cake should be incremented and the increment must be 
// canceled at R+1).

// So how to calculate the final heights? (Think a little bit).

// for(int i = 1 ; i ? C ; i++)
//    height[i] = height[i-1] + change[i];

// Simple !!

// Now which query we should remove?

// Let’s think about the outcome of removing one operation and canceling its effects. 
// All cakes between the L-th and the R-th (inclusive) will have their heights 
// decreased by 1. So after removing a certain operation, 
// the number of cakes with height K is equal to A+B+C

// A = number of cakes with a height equal to exactly K between the 1st cake and 
// the (L−1)th cake

// B = number of cakes with a height equal to exactly K+1 between the L-th cake and 
// the R-th cake

// C = number of cakes with a height equal to exactly K between the 
// (R+1)th cake and the last cake.

// So after calculating the heights resulting from applying the operations, we are 
// interested in only 2 heights. K and (K+1).

// Keep 2 prefix sum arrays, target[1...C], targetplus[1...,C]

// target[i] denotes the number of cakes with final height equal to K among the first 
// i cakes.

// targetplus[i] denotes the number of cakes with final height equal to K+1 among the 
// first i cakes.

// Now, it’s easy to check each operation and check the effect of removal and update 
// our answer accordingly. Check codes for more details.

// Complexity O(N+Q)


void solve(){
  ll n,k,cakes;cin>>n>>k;
  cakes=100000;
  ii query[cakes+5];
  vl h(cakes+5,0),target(cakes+5,0),targetplus(cakes+5,0);
  rep(i,1,n){
    int l,r;cin>>l>>r;
    query[i]={l,r};
    h[l]++;
    h[r+1]--;
  }
  rep(i,1,cakes){
    h[i]+=h[i-1];
    target[i]=target[i-1]+(h[i]==k);
    targetplus[i]=targetplus[i-1]+(h[i]==k+1);
  }   
  ll ans=0;
  rep(i,1,n){
    int l = query[i].ff;
    int r = query[i].ss;
    ans=max(ans,target[l-1]+targetplus[r]-targetplus[l-1]
                        +target[cakes]-target[r]);
  } 
  cout<<ans;
}
