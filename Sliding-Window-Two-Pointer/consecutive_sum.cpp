1004. Max Consecutive Ones III
---------------------------------
Given a binary array nums and an integer k, return the maximum number of consecutive 
1's in the array if you can flip at most k 0's.

 

Example 1:

Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.

Example 2:

Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.


int longestSubSeg(int a[], int n, int k)
{
    int cnt0 = 0;
    int l = 0;
    int max_len = 0;
 
    // i decides current ending point
    for (int i = 0; i < n; i++) {
        if (a[i] == 0)
            cnt0++;
        // If there are more 0's move
        // left point for current ending
        // point.
        while (cnt0 > k) {
            if (a[l] == 0)
                cnt0--;
            l++;
            debug(cnt0);
        }
 
        max_len = max(max_len, i - l + 1);
    }
 
    return max_len;
}


Intuition

Translation:
Find the longest subarray with at most K zeros.

Explanation

For each A[j], try to find the longest subarray.
If A[i] ~ A[j] has zeros <= K, we continue to increment j.
If A[i] ~ A[j] has zeros > K, we increment i (as well as j).

 int longestOnes(vector<int>& A, int K) {
        int i = 0, j;
        for (j = 0; j < A.size(); ++j) {
            if (A[j] == 0) K--;
            if (K < 0 && A[i++] == 0) K++;
        }
        return j - i;
    }