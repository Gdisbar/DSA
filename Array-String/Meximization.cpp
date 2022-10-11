A. Meximization # Codeforces Round #708 (Div. 2)
====================================================
// You are given an integer n and an array a1,a2,…,an. You should reorder the 
// elements of the array a in such way that the sum of MEX on prefixes 
// (i-th prefix is a1,a2,…,ai) is maximized.

// Formally, you should find an array b1,b2,…,bn, such that the sets of elements 
// of arrays a and b are equal (it is equivalent to array b can be found as an 
// array a with some reordering of its elements) and ∑i=(1,n)MEX(b1,b2,…,bi) is maximized.

// MEX of a set of nonnegative integers is the minimal non-negative integer 
// such that it is not in the set.

// For example, MEX({1,2,3})=0, MEX({0,1,2,4,5})=3

// If there exist multiple optimal answers you can find any.
// .
// Input

// 3 //t
// 7 //n
// 4 2 0 1 3 3 7 //a
// 5
// 2 2 8 6 9
// 1
// 0

// 0 1 2 3 4 7 3  //b
// 2 6 8 9 2 
// 0 

// Note

// In the first test case in the answer MEX for prefixes will be:

// MEX({0})=1
// MEX({0,1})=2
// MEX({0,1,2})=3
// MEX({0,1,2,3})=4
// MEX({0,1,2,3,4})=5
// MEX({0,1,2,3,4,7})=5
// MEX({0,1,2,3,4,7,3})=5

// The sum of MEX=1+2+3+4+5+5+5=25. It can be proven, that it is a maximum 
// possible sum of MEX on prefixes.

// To maximize the sum of MEX on prefixes we will use a greedy algorithm. 
// Firstly we put all unique elements in increasing order to get maximal MEX on 
// each prefix. It is easy to see that replacing any two elements after that makes 
// both MEX and sum of MEX less.
// In the end we put all elements that are not used in any order because MEX
// will not change and will still be maximal.

void solve() {
   int n;cin>>n;
   vi a(n);
   //a.clear();
   rep(i,0,n-1)cin>>a[i];
   sort(all(a));
   vi b;
   //b.clear();
   //store unique elements(increasing order) to get maximal MEX on each prefix
   //replacing any two elements after that makes both MEX and sum of MEX less.
   rep(i,0,n-1){ 
     if(i==0||a[i]!=a[i-1]) b.emplace_back(a[i]); 
     //emplace_back() doesn't make copies like push_back(),more efficient
   }
   //put elements that are not used in any order , we add below elements seen after | 

    // 0 1 2 3 4 7 
    // 2 6 8 9 
    // 0 

    // 0 1 2 3 4 7 | 3  //b
    // 2 6 8 9 | 2 
    // 0 
   rep(i,0,n-1){
     if(i>0&&a[i]==a[i-1]) b.emplace_back(a[i]);
   }
   each(x,b) cout<<x<<" ";
   cout<<endl;
   //cout.flush();
}


Function to return minimum MEX from all K-length subarrays
==================================================================
// Examples:

// Input: arr[] = {1, 2, 3}, K = 2
// Output: 1
// Explanation:
// All subarrays of length 2 are {1, 2}, {2, 3}.
// In subarray {1, 2}, the smallest positive integer which is not present is 3.
// In subarray {2, 3}, the smallest positive integer which is not present is 1.
// Therefore, the minimum of all the MEX for all subarrays of length K (= 2) is 1.

// Input: arr[] = {1, 2, 3, 4, 5, 6}, K = 3
// Output: 1


//Brute force generate all subarray of size k & find MEX --> (n-k+1)*k*n
// mx=a[i]
// mx=max(mx,a[i+j]) --> i=0 to n-k ,j=1 to k-1

// TC : n

// 
void minimumMEX(int arr[], int N, int K)
{
    // Stores element from [1, N + 1] which are not present in subarray
    set<int> s;
 
    // Store number 1 to N + 1 in set s
    for (int i = 1; i <= N + 1; i++)
        s.insert(i);
 
    // Find the MEX of K-length subarray starting from index 0
    for (int i = 0; i < K; i++)
        s.erase(arr[i]);
 
    int mex = *(s.begin()); //get 1st value in set i.e MEX of [0,K]
 
    // Find the MEX of all subarrays of length K by erasing arr[i] 
    //and inserting arr[i - K]
    for (int i = K; i < N; i++) {
        // all elements of current window can't be MEX for this window
        s.erase(arr[i]); 
        // last element of previous window is MEX for this window
        s.insert(arr[i - K]); 
 
        // Store first element of set as MEX of current subarray
        int firstElem = *(s.begin());
 
        // Update minimum MEX of all K length subarray
        mex = min(mex, firstElem);
    }
 
    
    cout << mex << ' ';
}


C1. k-LCM (easy version) 
===========================================================
// It is the easy version of the problem. The only difference is that in 
// this version k=3,You are given a positive integer n. 
// Find k positive integers a1,a2,…,ak, such that:
// 	a1+a2+…+ak=n
// 	LCM(a1,a2,…,ak)≤n/2
// We can show that for given constraints the answer always exists.


// 3
// 3 3 //n k
// 8 3
// 14 3

// 1 1 1
// 4 2 2
// 6 6 2


// If n is odd, then the answer is (1,⌊n/2⌋,⌊n/2⌋)
// If n is even, but is not a multiple of 4, then the answer is (n/2−1,n/2−1,2)
// If n is a multiple of 4, then the answer is (n/2,n/4,n/4). 

C2. k-LCM (hard version)
============================
// The only difference is that in this version 3≤k≤n.

// In this solution we will reuse the solution for k=3

// The answer will be 1,1,…,1 (k−3 times) and the solution a,b,c of the easy version 
// for n−k+3 i.e for k=3,n=n-k
// 	(1+1+…+1)+(a+b+c)=(k−3)+(n−k+3)=n
//     Also LCM(1,1,…,1,a,b,c)=LCM(a,b,c)≤n−k+3/2≤n/2
