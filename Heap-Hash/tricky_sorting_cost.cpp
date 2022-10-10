Tricky Sorting Cost
=======================
// Given an array arr[] of N elements containing first N positive integers. 
// You have to sort the integers in ascending order by the following operation. 
// eration is to pick an integer and place it at end or at start. Every such 
// operation increases cost by one. The task is to sort the array in the 
// minimum cost

// Example 1:

// Input: N = 3
// arr = {2, 1, 3}
// Output: 1
// Explaination: Place 1 at start.

// Example 2:

// Input: N = 4
// arr = {4, 3, 1, 2}
// Output: 2
// Explaination: First place 3 at end then 
// 4 at end.

// 3 1 2 8 9 4 7 5 6

// find min no of elements that needs to be moved --> n - max consecutive subseq

int sortingCost(int N, int arr[]){
       int mx=0; // longest increasing subsequnce length starting from arr[i]-1
       unordered_map<int,int>mp;
       for(int i=0;i<N;++i){
           int x = arr[i];
           //previous element exist 
           if(mp.find(x-1)!=mp.end())
            //subseq len made by previous + 1
                mp[x]=mp[x-1]+1; 
            else mp[x]=1; // prev doesn't exist,so new subseq
            mx=max(mx,mp[x]);
       }
       return N-mx;
    }

// Dynamic Programming Approach: Let DP[i] store the length of the 
// longest subsequence which ends with A[i]. For every A[i], if A[i]-1 
// is present in the array before i-th index, then A[i] will add to the 
// increasing subsequence which has A[i]-1. 
// Hence, DP[i] = DP[ index(A[i]-1) ] + 1. If A[i]-1 is not present in the 
// array before i-th index, then DP[i]=1 since the A[i] element forms a 
// subsequence which starts with A[i]. Hence, the relation for DP[i] is: 
 

//     If A[i]-1 is present before i-th index:  

//         DP[i] = DP[ index(A[i]-1) ] + 1

//     else: 

//         DP[i] = 1

int longestSubsequence(int a[], int n)
{
    // stores the index of elements
    unordered_map<int, int> mp;
 
    // stores the length of the longest
    // subsequence that ends with a[i]
    int dp[n];
    memset(dp, 0, sizeof(dp));
 
    int maximum = INT_MIN;
 
    // iterate for all element
    for (int i = 0; i < n; i++) {
 
        // if a[i]-1 is present before i-th index
        if (mp.find(a[i] - 1) != mp.end()) {
 
            // last index of a[i]-1
            int lastIndex = mp[a[i] - 1] - 1;
 
            // relation
            dp[i] = 1 + dp[lastIndex];
        }
        else
            dp[i] = 1;
 
        // stores the index as 1-index as we need to
        // check for occurrence, hence 0-th index
        // will not be possible to check
        mp[a[i]] = i + 1;
 
        // stores the longest length
        maximum = max(maximum, dp[i]);
    }
 
    return maximum;
}   