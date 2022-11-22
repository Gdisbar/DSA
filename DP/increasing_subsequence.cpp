491. Increasing Subsequences
=================================
// Given an integer array nums, return all the different possible increasing 
// subsequences of the given array with at least two elements. You may return 
// the answer in any order.

// The given array may contain duplicates, and two equal integers should also be 
// considered a special case of increasing sequence.

 

// Example 1:

// Input: nums = [4,6,7,7]
// Output: [[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]

// Example 2:

// Input: nums = [4,4,3,2,1]
// Output: [[4,4]]


class Solution {
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> seq;
        dfs(res, seq, nums, 0);
        return res;
    }
    
    void dfs(vector<vector<int>>& res, vector<int>& seq, vector<int>& nums,int pos) {
        if(seq.size() > 1) res.push_back(seq);
        unordered_set<int> hash; //avoid duplicate count of same element
        for(int i = pos; i < nums.size(); ++i) { //all the different possible subseq
            // nums[i] does'nt exist in hash + nums[i] is increasing
            if((seq.empty() || nums[i] >= seq.back()) && hash.find(nums[i]) == hash.end()) {
                seq.push_back(nums[i]);
                dfs(res, seq, nums, i + 1);
                seq.pop_back(); //backtrack
                hash.insert(nums[i]);
            }
        }
    }
};

    // def findSubsequences(self, nums: List[int]) -> List[List[int]]:
    //     ans=set()
    //     def dfs(pos,seq):
    //         if len(seq)>1:
    //             ans.add(tuple(seq))   
    //         last=seq[-1] if seq else -400
    //         for i in range(pos,len(nums)):
    //             if nums[i]>=last:
    //                 seq.append(nums[i])
    //                 dfs(i+1,seq)
    //                 seq.pop()
    //     dfs(0,[])
    //     return ans


Minimum number of increasing subsequences
============================================
// Given an array of integers of size N, you have to divide it into the 
// minimum number of “strictly increasing subsequences” 
// For example: let the sequence be {1, 3, 2, 4}, then the answer would be 2. 
// In this case, the first increasing sequence would be {1, 3, 4} and the 
// second would be {2}.

// Examples:

     
//     Input : arr[] = {1 3 2 4} 
//     Output: 2 
//     There are two increasing subsequences {1, 3, 4} and {2}
//     Input : arr[] = {4 3 2 1} 
//     Output : 4
//     Input : arr[] = {1 2 3 4} 
//     Output : 1
//     Input : arr[] = {1 6 2 4 3} 
//     Output : 3

// Min number of increasing subseq = length of longest decreasing subseq
// so it can be found in N*Log(N) time complexity in the same way as 
// longest increasing subsequence by multiplying all the elements with -1. 

//TC : n*log(n) , SC : n

arr[] = {1 6 2 4 3} 

i : 0, arr[i]=1, it=0 // last={1} , insert
i : 1, arr[i]=6, it=6 // last={1} -> last={6} , erase+insert
i : 2, arr[i]=2, it=6 // last={6} -> last={2,6} , insert
i : 3, arr[i]=4, it=4 //last={2,6} -> last={4,6} , erase+insert
i : 4, arr[i]=3, it=4 //last={4,6} -> last={3,4,6} , insert


int MinimumNumIncreasingSubsequences(int arr[], int n){
    // last element in each increasing subsequence found so far
    multiset<int> last;
    for (int i = 0; i < n; i++) {
        multiset<int>::iterator it = last.lower_bound(arr[i]);
        // iterator to the first element larger than or equal to arr[i]
        if (it == last.begin())
            // continue with this subsequence 
            // if all the elements in last larger
            // than or to arr[i] then insert it into last
            last.insert(arr[i]);
 
        else {
            it--;
            //start of new sub-sequence
            // the largest element smaller than arr[i] is the number
            // before *it which is it--
            last.erase(it); // erase the largest element smaller than arr[i]
            last.insert(arr[i]); // and replace it with arr[i]
        }
    }
    return last.size(); // our answer is the size of last
}

// // To search for correct position of num in array dp
// int search(vector<int>dp, int num){
 
//     // Initialise low,high and ans
//     int low = 0,high = dp.size() - 1;
//     int ans = -1;
//     while (low <= high){
 
//         // Get mid
//         int mid = low + ((high - low) / 2);
 
//         // If mid element is >=num search for left half
//         if (dp[mid] >= num){
//             ans = mid;
//             high = mid - 1;
//         }
         
//         else
//             low = mid + 1;
//     }
//     return ans;
// }
 
int longestDecrasingSubsequence(vector<int>A,int N){
    vector<int>dp(N+1,INT_MAX);  // min element of subsequence of length i
    dp[0] = INT_MIN;  
    // search for the correct position of number and insert in array
    for(int i = 0; i < N; i++){
        // search for the position
        //int index = search(dp, A[i]);
        int index = lower_bound(dp.begin(),dp.end(),A[i])-dp.begin();
        // update the dp array
        if (index != -1)
            dp[index] = min(dp[index], A[i]);
    }
     
    int Len = 0;
    for(int i = 1; i < N; i++){
        if (dp[i] != INT_MAX)
            Len = max(i, Len);
 
    }
    return Len;
}