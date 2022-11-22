Water Connection Problem 
===========================
// There are n houses and p water pipes in Geek Colony. 
// Every house has at most one pipe going into it and at most one pipe going out of it. 
// Geek needs to install pairs of tanks and taps in the colony according to the 
// following guidelines.  
// 1. Every house with one outgoing pipe but no incoming pipe gets a tank on its roof.
// 2. Every house with only one incoming and no outgoing pipe gets a tap.
// The Geek council has proposed a network of pipes where connections are denoted by 
// three input values: ai, bi, di denoting the pipe of diameter di from house ai to 
// house bi.
// Find a more efficient way for the construction of this network of pipes. 
// Minimize the diameter of pipes wherever possible.
// Note: The generated output will have the following format. The first line will 
// contain t, denoting the total number of pairs of tanks and taps installed. 
// The next t lines contain three integers each: house number of tank, house number 
// of tap, and the minimum diameter of pipe between them.


// Example 1:

// Input:
// n = 9, p = 6
// a[] = {7,5,4,2,9,3}
// b[] = {4,9,6,8,7,1}
// d[] = {98,72,10,22,17,66} 
// Output: 
// 3
// 2 8 22
// 3 1 66
// 5 6 10
// Explanation:

3 ----66----> 1
2 ----22 ---> 8
5 --- 72 ---> 9 --- 17 ---> 7 --- 98 ---> 4 --- 10 ---> 6  => 5 ---10---> 6

// Connected components are 
// 3->1, 5->9->7->4->6 and 2->8.
// Therefore, our answer is 3 
// followed by 2 8 22, 3 1 66, 5 6 10

class Solution
{
    public:
      map<int,pair<int,int>>m;
      int dfs(int i,int &dia)
      {
          if(!m[i].first)
          return i;
          dia=min(dia,m[i].second);
          return dfs(m[i].first,dia);
      }
    vector<vector<int>> solve(int n,int p,vector<int> a,vector<int> b,vector<int> d)
    {
        // code here
        vector<vector<int>>ans;
        vector<int>v(n+1,-1); //visited 
        for(int i=0;i<p;i++)
        {
            pair<int,int>pr(b[i],d[i]);
            m[a[i]]=pr;
            v[b[i]]=1;
        }
        for(int i=1;i<=n;i++)
        {
            if(v[i]==-1)
            {
                int in=i;
                int out;
                int dia=INT_MAX;
                out=dfs(i,dia);
                if(in!=out)
                {
                    vector<int>v;
                    v.push_back(in);
                    v.push_back(out);
                    v.push_back(dia);
                    ans.push_back(v);
                }
            }
        }
        return ans;
    }
};


// number of houses and number
// of pipes
int number_of_houses, number_of_pipes;
 
// Array rd stores the
// ending vertex of pipe
int ending_vertex_of_pipes[1100];
 
// Array wd stores the value
// of diameters between two pipes
int diameter_between_two_pipes[1100];
 
// Array cd stores the
// starting end of pipe
int starting_vertex_of_pipes[1100];
 
// Vector a, b, c are used
// to store the final output
vector<int> a;
vector<int> b;
vector<int> c;
 
int ans;
 
int dfs(int w)
{
    if (starting_vertex_of_pipes[w] == 0)
        return w;
    if (diameter_between_two_pipes[w] < ans)
        ans = diameter_between_two_pipes[w];
    return dfs(starting_vertex_of_pipes[w]);
}
 
// Function performing calculations.
void solve(int arr[][3])
{
    for (int i = 0; i < number_of_pipes; ++i) {
 
        int house_1 = arr[i][0], house_2 = arr[i][1],
            pipe_diameter = arr[i][2];
 
        starting_vertex_of_pipes[house_1] = house_2;
        diameter_between_two_pipes[house_1] = pipe_diameter;
        ending_vertex_of_pipes[house_2] = house_1;
    }
 
    a.clear();
    b.clear();
    c.clear();
 
    for (int j = 1; j <= number_of_houses; ++j)
 
        /*If a pipe has no ending vertex
        but has starting vertex i.e is
        an outgoing pipe then we need
        to start DFS with this vertex.*/
        if (ending_vertex_of_pipes[j] == 0
            && starting_vertex_of_pipes[j]) {
            ans = 1000000000;
            int w = dfs(j);
 
            // We put the details of component
            // in final output array
            a.push_back(j);
            b.push_back(w);
            c.push_back(ans);
        }
 
    cout << a.size() << endl;
    for (int j = 0; j < a.size(); ++j)
        cout << a[j] << " " << b[j] << " " << c[j] << endl;
}
 
// driver function
int main()
{
    number_of_houses = 9, number_of_pipes = 6;
 
    memset(ending_vertex_of_pipes, 0,sizeof(ending_vertex_of_pipes));
    memset(starting_vertex_of_pipes, 0,sizeof(starting_vertex_of_pipes));
    memset(diameter_between_two_pipes, 0,izeof(diameter_between_two_pipes));
 
    int arr[][3] = { { 7, 4, 98 }, { 5, 9, 72 }, { 4, 6, 10 },
            { 2, 8, 22 }, { 9, 7, 17 }, { 3, 1, 66 } };
 
    solve(arr);
    return 0;
}