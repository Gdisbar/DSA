
## Pattern 1 — Fixed Size Window

| Section | Details |
|---|---|
| Recognition | Exactly K • Size K • Average • Maximum/Minimum K |
| Core Idea | Only one element enters and one leaves per step — update the running sum by delta instead of recomputing. |
| Time / Space | O(n) / O(1) |
| Common Trap | Don't recompute the window sum from scratch each time. |

---

### 2461. Maximum Sum of Distinct Subarrays With Length K

You are given an integer array nums and an integer k. Find the maximum subarray sum of all the subarrays of nums that meet the following conditions:

- The length of the subarray is k, and
- All the elements of the subarray are distinct.

*Return the maximum subarray sum of all the subarrays that meet the conditions. If no subarray meets the conditions, return 0*.


**Example 1**:

    Input: nums = [1,5,4,2,9,9,9], k = 3
    Output: 15
    Explanation: The subarrays of nums with length 3 are:
    - [1,5,4] which meets the requirements and has a sum of 10.
    - [5,4,2] which meets the requirements and has a sum of 11.
    - [4,2,9] which meets the requirements and has a sum of 15.
    - [2,9,9] which does not meet the requirements because the element 9 is repeated.
    - [9,9,9] which does not meet the requirements because the element 9 is repeated.
    We return 15 because it is the maximum subarray sum of all the subarrays that meet the conditions

**Example 2**:

    Input: nums = [4,4,4], k = 3
    Output: 0
    Explanation: The subarrays of nums with length 3 are:
    - [4,4,4] which does not meet the requirements because the element 4 is repeated.
    We return 0 because no subarrays meet the conditions.
    
```python

"""
right - add curr element to window, calculate curr_sum
left - remove duplicate from window,when window_size == k remove one element
       to keep the process continue and record max_sum
Optimization : rather than calculating set(nums[left:right]) in every step
            use 2 pointer to reduce TC from O(n*k)
"""


def maximumSubarraySum(self, nums: List[int], k: int) -> int:
    if len(nums)<k:
        return 0
    window_set = set()
    curr_sum = 0
    max_sum = 0
    left = 0
    
    for right in range(len(nums)):
        # If nums[right] is already in the set, shrink window from left 
        # until the duplicate is removed
        while nums[right] in window_set:
            window_set.remove(nums[left])
            curr_sum -= nums[left]
            left += 1
        
        # Add current element to window
        window_set.add(nums[right])
        curr_sum += nums[right]
        
        # When window reaches size k, record max_sum and shift left pointer once
        if right - left + 1 == k:
            max_sum = max(max_sum, curr_sum)
            window_set.remove(nums[left])
            curr_sum -= nums[left]
            left += 1
            
    return max_sum

# java

import java.util.HashSet;
import java.util.Set;

public long maximumSubarraySum(int[] nums, int k) {
    if (nums == null || nums.length < k) {
        return 0;
    }

    Set<Integer> windowSet = new HashSet<>();
    long currSum = 0;
    long maxSum = 0;
    int left = 0;

    for (int right = 0; right < nums.length; right++) {
        while (windowSet.contains(nums[right])) {
            windowSet.remove(nums[left]);
            currSum -= nums[left];
            left++;
        }
        windowSet.add(nums[right]);
        currSum += nums[right];

        if (right - left + 1 == k) {
            maxSum = Math.max(maxSum, currSum);
            windowSet.remove(nums[left]);
            currSum -= nums[left];
            left++;
        }
    }

    return maxSum;
}

```

### 1052. Grumpy Bookstore Owner

There is a bookstore owner that has a store open for n minutes. You are given an integer array customers of length n where customers[i] is the number of the customers that enter the store at the start of the ith minute and all those customers leave after the end of that minute.

During certain minutes, the bookstore owner is grumpy. You are given a binary array grumpy where grumpy[i] is 1 if the bookstore owner is grumpy during the ith minute, and is 0 otherwise.

When the bookstore owner is grumpy, the customers entering during that minute are not satisfied. Otherwise, they are satisfied.

The bookstore owner knows a secret technique to remain not grumpy for minutes consecutive minutes, but this technique can only be used once.

*Return the maximum number of customers that can be satisfied throughout the day*.

 
**Example 1**:

    Input: customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], minutes = 3
    Output: 16
    Explanation:
    The bookstore owner keeps themselves not grumpy for the last 3 minutes.
    The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.

```python

"""
Include & Exclude is satisfied under different condiftion 
    - so window update is done in 2 step : adding & removing
"""

def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
    act_satisfy = sum(c for c, g in zip(customers, grumpy) if g == 0)
    extra_satisfy = 0
    max_extra_satisfy = 0

    for i in range(minutes):
        if grumpy[i]==1:
            extra_satisfy += customers[i]
    max_extra_satisfy = extra_satisfy

    for i in range(minutes,len(grumpy)):
        # Add new incoming element if owner was grumpy
        if grumpy[i]==1:
            extra_satisfy += customers[i]
        # Remove outgoing element if owner was grumpy
        if grumpy[i-minutes]==1:
            extra_satisfy -= customers[i-minutes]
        
        max_extra_satisfy = max(max_extra_satisfy,extra_satisfy)

    return act_satisfy + max_extra_satisfy

# java 

# len(customers) - customers..length
# max() - Math.max()
```