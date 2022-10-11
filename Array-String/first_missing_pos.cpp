41. First Missing Positive
=============================
// Given an unsorted integer array nums, return the smallest missing positive 
// integer.

// You must implement an algorithm that runs in O(n) time and uses constant 
// extra space.

 

// Example 1:

// Input: nums = [1,2,0]
// Output: 3
// Explanation: The numbers in the range [1,2] are all in the array.

// Example 2:

// Input: nums = [3,4,-1,1]
// Output: 2
// Explanation: 1 is in the array but 2 is missing.

// Example 3:

// Input: nums = [7,8,9,11,12]
// Output: 1
// Explanation: The smallest positive integer 1 is missing.

class Solution:
     def firstMissingPositive(self, nums: List[int]) -> int:
        one=False 
        # replace all out of range with one 
        for i in range(len(nums)):
            if nums[i] == 1:
                one = True
            if nums[i] < 1 or nums[i] > len(nums):
                nums[i] = 1
        if one == False:return 1  # 1 not present it''s the smallest +ve integer
        # map element to index - by making them -ve
        for i in range(len(nums)):
            idx = abs(nums[i])
            nums[idx-1] = -1*abs(nums[idx-1])
        # 1st +ve index is the answer if not found then last one of the range [1...len(nums)+1]
        # i.e len(nums)+1 is the answer
        for i in range(len(nums)):
            if nums[i] > 0:
                return i+1
        return 1+len(nums)
        