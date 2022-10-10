D. Same Differences # Codeforces Round #719 (Div. 3)
=========================================================
You are given an array a of n integers. Count the number of pairs of 
indices (i,j) such that i<j and a[j]−a[i]=j−i.

//Input
4
6
3 5 1 4 6 6
3
1 2 3
4
1 3 3 4
6
1 6 3 4 5 6

//Output
1
3
3
10

Let''s rewrite the original equality a bit:
a[j]−a[i]=j−i,
a[j]−j=a[i]−i

Let''s replace each ai
with b[i]=a[i]−i. Then the answer is the number of pairs (i,j) such that i<j 
and b[i]=b[j]. To calculate this value you can use map or sorting.


void solve() {
   int n;cin>>n;
   unordered_map<int,int> mp;
   ll cnt=0;
   rep(i,0,n-1){
     int x;cin>>x;
     x=x-i;
     cnt+=mp[x];
     mp[x]++;
   }
   cout<<cnt<<endl;
}