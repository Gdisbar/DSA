Minimum Cost to cut a board into squares
===========================================
A board of length m and width n is given, we need to break this board into 
m*n squares such that cost of breaking is minimum. cutting cost for each 
edge will be given for the board. In short, we need to choose such a sequence 
of cutting such that cost is minimized. 


X={2,1,3,1,4}
Y={4,1,2}

For above board optimal way to cut into square is:
Total minimum cost in above case is 42. It is 
evaluated using following steps.

Initial Value : Total_cost = 0
Total_cost = Total_cost + edge_cost * total_pieces

Cost 4 Horizontal cut         Cost = 0 + 4*1 = 4
Cost 4 Vertical cut        Cost = 4 + 4*2 = 12
Cost 3 Vertical cut        Cost = 12 + 3*2 = 18
Cost 2 Horizontal cut        Cost = 18 + 2*3 = 24
Cost 2 Vertical cut        Cost = 24 + 2*3 = 30
Cost 1 Horizontal cut        Cost = 30 + 1*4 = 34
Cost 1 Vertical cut        Cost = 34 + 1*4 = 38
Cost 1 Vertical cut        Cost = 38 + 1*4 = 42

X : horizontal cut : --|--|--|--|--  //i.e X[i++]>Y[j] 
                                                     
horizontal cut increase vertical width / no. of vertical pieces //vert++
total cost += horizontal cost * no. of vertical pieces (4)

Y : vertical cut :   |             // i.e X[i]<=Y[j++] 
                   ------     
	                 |
	               -----
	                 |
	               -----
	                 |

vertical cut increase horizontal width / no. of horizontal pieces //hzntl++
total cost += vertical cost * no. of horizontal pieces(3)   

int minimumCostOfBreaking(int X[], int Y[], int m, int n)
{
    int res = 0;
  
  	//we need to minimize # of cuts i.e maximize cost per cut
  	//Total_cost = Total_cost + edge_cost * total_pieces
    //  sort the horizontal cost in reverse order , 
    sort(X, X + m, greater<int>());
  
    //  sort the vertical cost in reverse order
    sort(Y, Y + n, greater<int>());
  
    //  initialize current width as 1
    int hzntl = 1, vert = 1;
  
    //  loop until one or both cost array are processed
    int i = 0, j = 0;
    while (i < m && j < n)
    {
        if (X[i] > Y[j]) //horizontal cut
        {
            res += X[i] * vert;
  
            //  increase current horizontal part count by 1
            hzntl++;
            i++;
        }
        else
        {
            res += Y[j] * hzntl;
  
            //  increase current vertical part count by 1
            vert++;
            j++;
        }
    }
  
    // loop for horizontal array, if remains
    int total = 0;
    while (i < m)
        total += X[i++];
    res += total * vert;
  
    // loop for vertical array, if remains
    total = 0;
    while (j < n)
        total += Y[j++];
    res += total * hzntl;
  
    return res;
}
  


CHOCOLA - Chocolate
=========================

// We are given a bar of chocolate composed of m*n square pieces. 
// One should break the chocolate into single squares. Parts of the chocolate 
// may be broken along the vertical and horizontal lines as indicated by the broken 
// lines in the picture.

// A single break of a part of the chocolate along a chosen vertical or horizontal 
// line divides that part into two smaller ones. Each break of a part of the 
// chocolate is charged a cost expressed by a positive integer. This cost does 
// not depend on the size of the part that is being broken but only depends on the 
// line the break goes along. Let us denote the costs of breaking along consecutive 
// vertical lines with x1, x2, ..., xm-1 and along horizontal lines 
// with y1, y2, ..., yn-1.

// The cost of breaking the whole bar into single squares is the sum of the 
// successive breaks. One should compute the minimal cost of breaking the whole 
// chocolate into single squares.

// For example, if we break the chocolate presented in the picture first along the 
// horizontal lines, and next each obtained part along vertical lines then the cost 
// of that breaking will be y1+y2+y3+4*(x1+x2+x3+x4+x5).

// Task

// Write a program that for each test case:

//     Reads the numbers x1, x2, ..., xm-1 and y1, y2, ..., yn-1
//     Computes the minimal cost of breaking the whole chocolate into 
//     single squares, writes the result.

// Input

// One integer in the first line, stating the number of test cases, followed by a 
// blank line. There will be not more than 20 tests.

// For each test case, at the first line there are two positive integers m and n 
// separated by a single space, 2 <= m,n <= 1000. In the successive m-1 lines there 
// are numbers x1, x2, ..., xm-1, one per line, 1 <= xi <= 1000. In the successive 
// n-1 lines there are numbers y1, y2, ..., yn-1, one per line, 1 <= yi <= 1000.

// The test cases will be separated by a single blank line.
// Output

// For each test case : write one integer - the minimal cost of breaking the whole 
// chocolate into single squares.

// Example

// Input:
// 1

// 6 4
// 2
// 1
// 3
// 1
// 4
// 4
// 1
// 2

// Output:
// 42


int main(void)
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
#ifndef ONLINE_JUDGE
   freopen("input.txt", "r", stdin);
   freopen("output.txt", "w", stdout);
   freopen("error.txt", "w", stderr);
#endif
   ll tc,T=1;
   cin>>tc;
   //cout<<endl;
  // precompute();
   ll x[1001],y[1001],dp[1001][1001];
   while(tc--){
   // solve();
   //  cout<<'\n';

       ll n,m;cin>>m>>n;
       m--;
       n--;
       rep(i,1,m)cin>>x[i];
       rep(i,1,n)cin>>y[i];
       sort(x+1,x+1+m,greater<ll>());
       sort(y+1,y+1+n,greater<ll>());
       dp[0][0]=0;
       rep(i,1,m) dp[i][0]=x[i]+dp[i-1][0]; // 1st horizontal cut
       rep(i,1,n) dp[0][i]=y[i]+dp[0][i-1]; //1st vertical cut
       rep(i,1,m){
         rep(j,1,n){
            ll horz=dp[i-1][j] + x[i]*(j+1); // increase in vertical width
            ll vert=dp[i][j-1] + y[j]*(i+1); //increase in horizontal width
            dp[i][j]=min(horz,vert);
         }
       }
       cout<<dp[m][n]<<endl;
  } 
}