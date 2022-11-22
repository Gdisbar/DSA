152. Maximum Product Subarray
=================================
Given an integer array nums, find a contiguous non-empty subarray 
within the array that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.


//Kadane

int maxProduct(vector<int> A) {
        int n = A.size(), res = A[0], pref = 0, sufx = 0;
        for (int i = 0; i < n; i++) {
            pref =  (pref ? pref : 1) * A[i];
            sufx =  (sufx ? sufx : 1) * A[n - 1 - i];
            res = max(res, max(pref, sufx));
        }
        return res;
    }