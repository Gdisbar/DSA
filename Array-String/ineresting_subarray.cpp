B. Interesting Subarray | Good Bye 2019
=========================================
// For an array a of integers let''s denote its maximal element as max(a), and 
// minimal as min(a). We will call an array a of k integers interesting if 
// max(a)−min(a)≥k. For example, array [1,3,4,3] isn''t interesting as 
// max(a)−min(a)=4−1=3<4 while array [7,3,0,4,3] is as max(a)−min(a)=7−0=7≥5
// You are given an array a of n integers. Find some interesting nonempty subarray 
// of a, or tell that it doesn''t exist.

// An array b is a subarray of an array a if b can be obtained from a
// by deletion of several (possibly, zero or all) elements from the beginning and 
// several (possibly, zero or all) elements from the end. In particular, an array 
// is a subarray of itself.

// output "YES" in a separate line. In the next line, output two integers 
// l and r (1≤l≤r≤n) — bounds of the chosen subarray. If there are multiple answers, 
// print any.

// Input
// 3 --> test case

// 5  --> n
// 1 2 3 4 5 --> array
// NO

// 4
// 2 0 1 9
// YES
// 1 9 ---> l, r

// 2
// 2019 2020
// NO


// Note
// In the second test case of the example, one of the interesting subarrays is 
// a=[2,0,1,9]: max(a)−min(a)=9−0=9≥4.

// We will show that if some interesting nonempty subarray exists, then also exists 
// some interesting subarray of length 2. Indeed, let a[l..r] be some interesting 
// nonempty subarray, let amax be the maximal element, amin — minimal, without loss 
// of generality, max>min. Then amax−amin≥r−l+1≥max−min+1, or 
// (amax−amax−1)+(amax−1−amax−2)+⋯+(amin+1−amin)≥max−min+1, so at least one of the 
// terms had to be ≥2. Therefore, for some i holds a[i+1]−a[i]≥2, so subarray [i,i+1]

// is interesting!

// Therefore, the solution is as follows: for each i
// from 1 to n−1 check if |a[i+1]−a[i]|≥2 holds. If this is true for some i, we have 
// found an interesting subarray of length 2, else such subarray doesn''t exist.

void solve(){
   int n;cin>>n;
   
   vi a(n);
   rep(i,0,n-1){
     cin>>a[i];
   }
   rep(i,0,n-2){
     if(abs(a[i+1]-a[i])>=2){
        cout<<"YES"<<endl<<i+1<<" "<<i+2;
        return;
     }
   }
   cout<<"NO";
}