1031. Maximum Sum of Two Non-Overlapping Subarrays
======================================================
// Given an integer array nums and two integers firstLen and secondLen, return 
// the maximum sum of elements in two non-overlapping subarrays with lengths 
// firstLen and secondLen.

// The array with length firstLen could occur before or after the array with 
// length secondLen, but they have to be non-overlapping.

// A subarray is a contiguous part of an array.


// Example 1:

// Input: nums = [0,6,5,2,2,5,1,9,4], firstLen = 1, secondLen = 2
// Output: 20
// Explanation: One choice of subarrays is [9] with length 1, and [6,5] with length 2.

// Example 2:

// Input: nums = [3,8,1,3,2,1,8,9,0], firstLen = 3, secondLen = 2
// Output: 29
// Explanation: One choice of subarrays is [3,8,1] with length 3, and [8,9] with 
// length 2.

// Example 3:

// Input: nums = [2,1,5,6,0,9,5,0,3,8], firstLen = 4, secondLen = 3
// Output: 31
// Explanation: One choice of subarrays is [5,6,0,9] with length 4, and [0,3,8] with 
// length 3.

// Similar to Best Time to Buy and Sell Stock III, but instead of maximum profit, 
// we track maximum sum of N elements.

// Left-to-right, track the maximum sum of L elements in left. 
// Right-to-left, track the maximum sum of M elements in right.

// Then, find the split point where left[i] + right[i] gives us the maximum sum.

// Note: we need to do it twice for (L, M) and (M, L).

int maxTwoNoOverlap(vector<int>& A, int L, int M, int sz, int res = 0) {
  vector<int> left(sz + 1), right(sz + 1);
  for (int i = 0, j = sz - 1, s_r = 0, s_l = 0; i < sz; ++i, --j) {
    s_l += A[i], s_r += A[j];
    left[i + 1] = max(left[i], s_l);
    right[j] = max(right[j + 1], s_r);
    if (i + 1 >= L) s_l -= A[i + 1 - L]; //upper bound i-M
    if (i + 1 >= M) s_r -= A[j + M - 1]; //lower bound i-M
  }
  for (auto i = 0; i < A.size(); ++i) {
    res = max(res, left[i] + right[i]);
  }
  return res;
}
int maxSumTwoNoOverlap(vector<int>& A, int L, int M) {
  // L is ahead of M and M is ahead L
  return max(maxTwoNoOverlap(A, L, M, A.size()), maxTwoNoOverlap(A, M, L, A.size()));
}



Q4: Why you use the maxL and sumM to calculate the ans and not maxM as well?
A4: That will result overlap of L-subarray and M-subarray. 
We traverse input array A from left to right, and the value of maxM could occur 
at part or whole of current L-subarray. E.g.,
A = [1,1,3,2,1], L = 2, M = 1
When L-subarray = [3,2] and M-subarray = [1], then maxM = 3, and 
the 3 is from L-subarray [3,2].

In contrast, maxL will never occur at part or whole of current M-subarray 
hence will NOT cause overlap, because L-subarray has not included any element 
in M-subarray yet.


Scan the prefix sum array from index L + M, which is the first possible position;
update the max value of the L-length subarray; then update max value of the sum 
of the both;
we need to swap L and M to scan twice, since either subarray can occur before 
the other.
In private method, prefix sum difference p[i - M] - p[i - M - L] is L-length 
subarray from index i - M - L to i - M - 1, and p[i] - p[i - M] is M-length 
subarray from index i - M to i - 1.

    def maxSumTwoNoOverlap(self, A: List[int], L: int, M: int) -> int:
        
        def maxSum(L:int, M:int) -> int:
            maxL = ans = 0
            for i in range(L + M, len(prefixSum)):
                maxL = max(maxL, prefixSum[i - M] - prefixSum[i - L - M])
                ans = max(ans, maxL + prefixSum[i] - prefixSum[i - M])
            return ans
        
        prefixSum = [0] * (len(A) + 1)
        for i, a in enumerate(A):
            prefixSum[i + 1] = prefixSum[i] + a
        return max(maxSum(L, M), maxSum(M, L)

def maxSumTwoNoOverlap(self, A: List[int], L: int, M: int) -> int:
        
        def maxSum(L:int, M:int) -> int:
            sumL = sumM = 0
            for i in range(0, L + M):
                if i < L:
                    sumL += A[i]
                else:
                    sumM += A[i]    
            maxL, ans = sumL, sumL + sumM
            for i in range(L + M, len(A)):
                sumL += A[i - M] - A[i - L - M]
                maxL = max(maxL, sumL)
                sumM += A[i] - A[i - M]
                ans = max(ans, maxL + sumM)
            return ans
        
        return max(maxSum(L, M), maxSum(M, L)) 


def maxSumTwoNoOverlap(self, nums: List[int], firstLen: int, secondLen: int) -> int:
        n=len(nums)
        pref=[0]*(n+1)
        for i in range(1,n+1):
            pref[i]=pref[i-1]+nums[i-1]
        
        def helper(l,r):
            lmx,rmx,mx=pref[l],pref[r+l]-pref[l],0
            mx=lmx+rmx
            for i in range(l+1,n-r+1):
                lmx=max(lmx,pref[i]-pref[i-l])
                rmx=pref[r+i]-pref[i]
                mx=max(mx,lmx+rmx)
            return mx
        
        l=helper(firstLen,secondLen)
        r=helper(secondLen,firstLen)
        return max(l,r)