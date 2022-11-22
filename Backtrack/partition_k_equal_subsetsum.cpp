698. Partition to K Equal Sum Subsets
=======================================
// Given an integer array nums and an integer k, return true if it is possible 
// to divide this array into k non-empty subsets whose sums are all equal.


// Example 1:

// Input: nums = [4,3,2,3,5,2,1], k = 4
// Output: true
// Explanation: It is possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) 
// with equal sums.

// Example 2:

// Input: nums = [1,2,3,4], k = 3
// Output: false


// To tell whether there are exactly k subsets with equal subset sum sum/k, we 
// may start from

//     each subset: for each subset, put any numbers inside
//     each number: for each number, put it into any subset

// 1. Starting from each subset

// we can take it as a nested recursion. The graph below shows the control 
// flow (not accurate):

// Outer recursion on k subsets:
// Base case: k == 0
// Recursive case: k > 0 
// 				// Inner recursion on individual subset
// 				Base case: curSubsetSum == targetSubsetSum (return to outer 
// 					recursion)
// 				Recursive case: curSubsetSum < targetSubsetSum


def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    
    nums_sum = sum(nums)
    if nums_sum % k != 0:
        return False
    subset_sum = nums_sum / k
    nums.sort(reverse=True)
    visited = [False] * len(nums)
    
    def can_partition(k, cur_sum=0, index=0):
        if k == 1:
            return True
        
        if cur_sum == subset_sum:
            return can_partition(k - 1)
        
        for i in range(index, len(nums)):
            if not visited[i] and cur_sum + nums[i] <= subset_sum:
                visited[i] = True
                if can_partition(k, cur_sum=cur_sum + nums[i], index=i + 1):
                    return True
                visited[i] = False
        return False 
    
    return can_partition(k)

// 2. Starting from each number


// Recursion on len(nums) numbers:
// Base case: j == len(nums)
// Recursive case: j < len(nums)


    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        
        nums_sum = sum(nums)
        if nums_sum % k != 0:
            return False
        subset_sum = nums_sum / k
        
        ks = [0] * k
        nums.sort(reverse=True)
        
        def can_partition(j):
            if j == len(nums):
                for i in range(k):
                    if ks[i] != subset_sum:
                        return False
                return True
            for i in range(k):
                if ks[i] + nums[j] <= subset_sum:
                    ks[i] += nums[j]
                    if can_partition(j + 1):
                        return True
                    ks[i] -= nums[j]
            return False

        return can_partition(0)
