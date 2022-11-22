GERGOVIA - Wine trading in Gergovia
====================================
// Gergovia consists of one street, and every inhabitant of the city is a 
// wine salesman. Everyone buys wine from other inhabitants of the city. 
// Every day each inhabitant decides how much wine he wants to buy or sell. 
// Interestingly, demand and supply is always the same, so that each inhabitant 
// gets what he wants.

// There is one problem, however: Transporting wine from one house to another 
// results in work. Since all wines are equally good, the inhabitants of 
// Gergovia don't care which persons they are doing trade with, they are 
// only interested in selling or buying a specific amount of wine.

// In this problem you are asked to reconstruct the trading during one day in 
// Gergovia. For simplicity we will assume that the houses are built along 
// a straight line with equal distance between adjacent houses. Transporting 
// one bottle of wine from one house to an adjacent house results in one unit 
// of work.

// Input

// The input consists of several test cases.

// Each test case starts with the number of inhabitants N (2 ≤ N ≤ 100000).

// The following line contains n integers ai (-1000 ≤ ai ≤ 1000).

// If ai ≥ 0, it means that the inhabitant living in the ith house wants to 
// buy ai bottles of wine. If ai < 0, he wants to sell -ai bottles of wine.

// You may assume that the numbers ai sum up to 0.

// The last test case is followed by a line containing 0.
// Output

// For each test case print the minimum amount of work units needed so that 
// every inhabitant has his demand fulfilled.

// Example

// Input:
// 5
// 5 -4 1 -3 1
// 6
// -1000 -1000 -1000 1000 1000 1000
// 0

// Output:
// 9
// 9000

// sm = keep track of wine we have bought or sold
// cost = rather than moving min(buyer,seller) to target 
// i.e cost = min(buyer,seller)*(distance btw them)
// we take carrying cost of wine at each step if we buy new stock along the inhabitant
// ways cost increase by that amount but if we sell we get an updated carrying cost

// i    0      1      2      3       4       5

// sm  -1000  -2000  -3000   -2000   -1000   0
// cost 1000   3000   6000    8000    9000   9000


while(1){
    int n;cin>>n;
    if(n==0)break;
    vl a(n);
    each(x,a)cin>>x;
    ll cost=0,sm=0;
    rep(i,0,n-1){
        sm+=a[i]; 
        cost+=abs(sm); 
    }
    cout<<cost<<endl;
}