1671. Minimum Number of Removals to Make Mountain Array
==========================================================
// You may recall that an array arr is a mountain array if and only if:

//     arr.length >= 3
//     There exists some index i (0-indexed) with 0 < i < arr.length - 1 such 
//     that:
//         arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
//         arr[i] > arr[i + 1] > ... > arr[arr.length - 1]

// Given an integer array nums, return the minimum number of elements to remove 
// to make nums a mountain array.

 

// Example 1:

// Input: nums = [1,3,1] //1 2 2
// Output: 0
// Explanation: The array itself is a mountain array so we do not need to 
// remove any elements.

// Example 2:

// Input: nums = [2,1,1,5,6,2,3,1] //1 1 1 2 3 3 3 3 
// Output: 3
// Explanation: One solution is to remove the elements at indices 0, 1, and 5, 
// making the array nums = [1,5,6,3,1].

int minimumMountainRemovals(vector<int>& nums) {
       int n = nums.size(),mx=-1;
       vector<int> lis(n,1),lds(n,1);
       for(int i=0;i<n;++i){
           for(int j=0;j<i;++j){
               if(nums[i]>nums[j]&&lis[j]+1>lis[i])
                   lis[i]=lis[j]+1;
           }
       }
       for(int i=n-2;i>=0;--i){
           for(int j=n-1;j>i;--j){
               if(nums[i]>nums[j]&&lds[j]+1>lds[i])
                   lds[i]=lds[j]+1;
           }
           if(lis[i]>1&&lds[i]>1)
             mx=max(mx,lis[i]+lds[i]-1);
       }
       
        return n-mx;
    }
    
// TC : nlogn , SC : n
int minimumMountainRemovals(vector<int>& n) {
        int res = INT_MAX, sz = n.size();
        vector<int> l, r, dp(sz);
        for (int i = 0; i < sz; ++i) {
            auto it = lower_bound(begin(l), end(l), n[i]);
            if (it == l.end())
                l.push_back(n[i]);
            else 
                *it = n[i];
            dp[i] = l.size();
        }
        // for (auto n : dp)
        //     cout << n << " ";
        // cout << endl;
        for (int i = n.size() - 1; i > 0; --i) {
            auto it = lower_bound(begin(r), end(r), n[i]);
            if (it == r.end())
                r.push_back(n[i]);
            else 
                *it = n[i];
            if (dp[i] > 1 && r.size() > 1)
                res = min(res, sz - dp[i] - (int)r.size() + 1);
        }
        return res;
    }


def minimumMountainRemovals(self, nums):
        def LIS(nums):
            dp = [10**10] * (len(nums) + 1)
            lens = [0]*len(nums)
            for i, elem in enumerate(nums): 
                lens[i] = bisect_left(dp, elem) + 1
                dp[lens[i] - 1] = elem 
            return lens
        
        l1, l2 = LIS(nums), LIS(nums[::-1])[::-1]
        ans, n = 0, len(nums)
        for i in range(n):
            if l1[i] >= 2 and l2[i] >= 2:
                ans = max(ans, l1[i] + l2[i] - 1)
                
        return n - ans