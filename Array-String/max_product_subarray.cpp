152. Maximum Product Subarray
================================
// Given an integer array nums, find a contiguous non-empty subarray within the 
// array that has the largest product, and return the product.

// The test cases are generated so that the answer will fit in a 32-bit integer.

// A subarray is a contiguous subsequence of the array.

 

// Example 1:

// Input: nums = [2,3,-2,4]
// Output: 6
// Explanation: [2,3] has the largest product 6.

// Example 2:

// Input: nums = [-2,0,-1]
// Output: 0
// Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

int maxProduct(vector<int>& nums) {
        int mx_end=nums[0];
        int mn_end=nums[0];
        int mx_so_far=nums[0];
        for(int i=1;i<nums.size();++i){
            int tmp = max({nums[i],nums[i]*mx_end,nums[i]*mn_end});
            mn_end= min({nums[i],nums[i]*mx_end,nums[i]*mn_end});
            mx_end=tmp;
            mx_so_far=max(mx_so_far,mx_end);
        }
        return mx_so_far;
    }

// First, if there''s no zero in the array, then the subarray with maximum product 
// must start with the first element or end with the last element. And therefore, 
// the maximum product must be some prefix product or suffix product. So in this 
// solution, we compute the prefix product A and suffix product B, and simply return 
// the maximum of A and B.

// What if there are zeroes in the array? Well, we can split the array into several 
// smaller ones. That''s to say, when the prefix product is 0, we start over and 
// compute prefix profuct from the current element instead. And this is exactly 
// what A[i] *= (A[i - 1]) or 1 does.


    int maxProduct(vector<int> A) {
        int n = A.size(), res = A[0], l = 0, r = 0;
        for (int i = 0; i < n; i++) {
            l =  (l ? l : 1) * A[i];
            r =  (r ? r : 1) * A[n - 1 - i];
            res = max(res, max(l, r));
        }
        return res;
    }


    def maxProduct(self, A):
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(A + B)