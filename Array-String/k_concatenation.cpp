K-Concatenation , Problem Code:KCON
======================================
// You are given an array A with size N (indexed from 0) and an integer K. Let''s 
// define another array B with size N * K as the array that''s formed by 
// concatenating K copies of array A.

// For example, if A = {1, 2} and K = 3, then B = {1, 2, 1, 2, 1, 2}.

// You have to find the maximum subarray sum of the array B. Fomally, you should 
// compute the maximum value of Bi + Bi+1 + Bi+2 + ... + Bj, where 0 ≤ i ≤ j < N · K.

// Input: 
// 2

// 2 3
// 1 2

// 3 2
// 1 -2 1
// Output:
// 9
// 2

//Brute force - construct B array + apply Kadanae's algorithm but forming B array 
//take 10^5*10^5 space & time

//Optimised Method://SOE : Sum Of Elements
// A= {3, 2, -1} , B = {3, 2, -1, 3, 2, -1, 3, 2, -1} ,K=3
// (Sum of elements in B) = (Sum of elements in A)*K = 4*3 = 12, but if we omit last term 
// of B we get Sum = 13 For this we need prefix & suffix calculations.

// A ={-1, -2, 8, 9, 10, -2, -1} ,K=10.
// B = {A1, A2, A3, A4, A5, A6, A7, A8, A9, A10}. 
// (sum of elements in A)*K = 270. But if you omit the first two elements in A1 
// and the last two elements in A10, You will get the Maximum SubArray as 276. 
// So here we can check whether it is possible to omit some initial elements in A1 
// and some Final elements in A10. 
// prefix and suffix variables for that to calculate the sum of A1 and A10 specifically 
// answer={prefix+SOE(A2)+...+SOE(A9)+suffix}={prefix+SOE(A)*(k-2)+suffix}. 


// A={12, -200, 12},K=3. Now the total sum becomes -ve. 
// B={12, -200, 12, 12, -200, 12, 12, -200, 12} 
// i.e {A1, A2, A3}. Here the prefix will be 12 and suffix will also be 12. so 
// the answer will be 24 which is the sum that exists at indexes 2 and 3, and 
// indexes 5 and 6.

void solve() {
   ll n,k;cin>>n>>k;
   vl a(n);
   each(x,a)cin>>x;
   ll ans=*max_element(all(a)),s=0;
   rep(i,0,n-1){
     s+=a[i];
     if(s<0)s=0;
     else ans=max(ans,s); //if ans==s ,then whole array is taken
   }
   
   if(k>1){
     ll prf=0,suf=0,p=0,s=0;
     rep(i,0,n-1){
       p+=a[i];
       s+=a[n-i-1];
       prf=max(prf,p);
       suf=max(suf,s);
     }
     
     if(prf+suf>0){
         s=max(accumulate(all(a),0ll),0ll);
         ans=max(ans,prf+suf+s*(k-2));
     }
     
   }
   cout<<ans;
}

// Different approach 
// It can be solved using Arithmetic Progression…
// Just Find the value of Array A for K=1 , K=2, K=3 and see the pattern…
// let’s say for k=1 answer is X1
// for k=2 answer is X2
// for k=3 answer is X3
// if k==1 print X1
// if k==2 print X2
// if k>2 print x2+(k-2)*(x3-x2)