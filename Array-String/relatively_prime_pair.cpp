B. Relatively Prime Pairs #Educational Codeforces Round 51 (Rated for Div. 2)
=================================================================================  
You are given a set of all integers from l to r inclusive, l<r, (r−l+1)≤3⋅105 and (r−l)
is always odd.You want to split these numbers into exactly (r−l+1)/2 pairs in such 
a way that for each pair (i,j) the greatest common divisor of i and j is equal to 1. 
Each number should appear in exactly one of the pairs.Print the resulting pairs or 
output that no solution exists. If there are multiple solutions, print any of them.


// Numbers with the difference of 1 are always relatively prime. 
// TC : r-l
1 8

1 2
3 4
5 6
7 8

ll k=(r-l+1)/2;
bool f=true;
vector<pair<ll,ll>> p;
if(k>0&&(r-l)%2==1){
  while(k>0&&l<r){
      p.pb({l,l+1});
      l=l+2;
  }
}
else f=false;