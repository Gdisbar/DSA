
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


## Pattern 2 — Variable Window

| Section | Details |
|---|---|
| Recognition | Longest • Shortest • At Most • Without Repeating |
| Core Idea | `while(right<n){ expand; while(invalid) shrink; update answer; }` |
| Common Trap | Never shrink blindly — always know *which* invariant you're restoring. |

**What Is an Invariant?**
A property that stays true as the window moves — e.g. "every character appears once" (Problem 3). Expand until it breaks, shrink until it's true again; never restart from scratch.

**Common Invariants:** Sum ≤ K • Distinct ≤ K • Frequency ≤ Limit • Length == K • Zero Count ≤ K • Cost ≤ Budget — each becomes its own pattern below.

---

### 3. Longest Substring Without Repeating Characters

Given a string s, find the length of the longest substring without duplicate characters.

**Example 1**:

    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3. Note that "bca" and "cab" are also correct answers.

**Example 2**:

    Input: s = "bbbbb"
    Output: 1
    Explanation: The answer is "b", with the length of 1.

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    seen = set()
    max_length = 0
    l = 0

    for r in range(len(s)):
        while s[r] in seen:
            seen.remove(s[l])
            l += 1
        seen.add(s[r])
        # to get the string we can store index r from set
        max_length = max(max_length, r - l + 1)

    return max_length
```

```java
public int lengthOfLongestSubstring(String s) {
    // Store the last seen index of each character (+1 offset)
    int[] lastSeen = new int[128];
    Arrays.fill(lastSeen, -1);
    
    int maxLength = 0;
    int l = 0;

    for (int r = 0; r < s.length(); r++) {
        char currChar = s.charAt(r);
        
        // If character was seen inside the current window, jump left pointer
        if (lastSeen[currChar] >= l) {
            l = lastSeen[currChar] + 1;
        }

        lastSeen[currChar] = r;
        maxLength = Math.max(maxLength, r - l + 1);
    }

    return maxLength;
}
```

### 76. Minimum Window Substring

Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.


**Example 1**:

    Input: s = "ADOBECODEBANC", t = "ABC"
    Output: "BANC"
    Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

**Example 2**:

    Input: s = "a", t = "aa"
    Output: ""
    Explanation: Both 'a's from t must be included in the window.
    Since the largest window of s only has one 'a', return empty string.

```python

def minWindow(self, s: str, t: str) -> str:
    if not s or not t or len(t) > len(s): 
        return ""
    t_counts = Counter(t)
    window_counts = {}
    
    have, need = 0, len(t_counts)
    res_len, res_bounds = float("inf"), [-1, -1]
    l = 0

    for r, char in enumerate(s):
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in t_counts and window_counts[char] == t_counts[char]:
            have += 1

        # Shrink from left once target condition is met
        while have == need:
            if (r - l + 1) < res_len:
                res_len = r - l + 1
                res_bounds = [l, r]

            left_char = s[l]
            window_counts[left_char] -= 1
            if left_char in t_counts and window_counts[left_char] < t_counts[left_char]:
                have -= 1
            l += 1
    start, end = res_bounds
    return s[start : end + 1] if res_len != float("inf") else ""
```

## Pattern 3 — Frequency Window

| Section | Details |
|---|---|
| Recognition | Characters • Occurrences • Anagram • Permutation |
| Core Idea | Maintain a frequency map inside the window; window is valid when it matches the target frequency. |
| Common Trap | Known alphabet? `int freq[26]` beats HashMap. |

---


### 567. Permutation in String

Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.


**Example 1**:

    Input: s1 = "ab", s2 = "eidbaooo"
    Output: true
    Explanation: s2 contains one permutation of s1 ("ba").

**Example 2**:

    Input: s1 = "ab", s2 = "eidboaoo"
    Output: false

```python
def checkInclusion(self, s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False

    freq_s1 = [0] * 26
    freq_s2 = [0] * 26
    for char in s1:
        freq_s1[ord(char) - ord('a')] += 1

    l = 0
    for r in range(len(s2)):
        # Add current character to window
        freq_s2[ord(s2[r]) - ord('a')] += 1
        # When window size exceeds len(s1), shrink from left
        if r - l + 1 > len(s1):
            freq_s2[ord(s2[l]) - ord('a')] -= 1
            l += 1
        # Check if current window matches s1 permutation
        if r - l + 1 == len(s1):
            if freq_s1 == freq_s2:
                return True
                
    return False

```

### 438. Find All Anagrams in a String

Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.


**Example 1**:

    Input: s = "abab", p = "ab"
    Output: [0,1,2]
    Explanation:
    The substring with start index = 0 is "ab", which is an anagram of "ab".
    The substring with start index = 1 is "ba", which is an anagram of "ab".
    The substring with start index = 2 is "ab", which is an anagram of "ab".

```python
# Exact same like 567 only if match found store 'l' in result = []
```

## Pattern 4 — Sum Window

| Section | Details |
|---|---|
| Recognition | Positive numbers • Continuous sum • Minimum length |
| Core Idea | With all-positive elements, expanding always increases the sum and shrinking always decreases it — the monotonic property Sliding Window needs. |
| Representative Problems | 209, 713, 1658 |
| Common Trap | Negative numbers present? This pattern breaks — use Prefix Sum (Pattern 11) instead. |

---

### 209. Minimum Size Subarray Sum

Given an array of positive integers nums and a positive integer target, return the minimal length of a subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.


**Example 1**:

    Input: target = 7, nums = [2,3,1,2,4,3]
    Output: 2
    Explanation: The subarray [4,3] has the minimal length under the problem constraint.

**Example 2**:

    Input: target = 11, nums = [1,1,1,1,1,1,1,1]
    Output: 0

```python
# add to current_sum while sm >= target: update window,current_sum,l
# also add the Example 2 case
```

### 713. Subarray Product Less Than K

Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.

 
**Example 1**:

    Input: nums = [10,5,2,6], k = 100
    Output: 8
    Explanation: The 8 subarrays that have product less than 100 are:
    [10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
    Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.


```python
"""
a valid window from index l to r contains r - l + 1 
valid contiguous subarrays that all end at index r
"""
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    cur_prod = 1
    l = 0
    subarr_count = 0
    for r in range(len(nums)):
        cur_prod *=nums[r]
        while cur_prod >= k and l < r:
            cur_prod //=nums[l]
            l+=1
        # a valid window from index l to r contains r - l + 1 
        # valid contiguous subarrays that all end at index r
        if cur_prod < k:
            subarr_count +=r-l+1

    return subarr_count
```

### 1658. Minimum Operations to Reduce X to Zero

You are given an integer array nums and an integer x. In one operation, you can either remove the leftmost or the rightmost element from the array nums and subtract its value from x. Note that this modifies the array for future operations.

Return the minimum number of operations to reduce x to exactly 0 if it is possible, otherwise, return -1.


**Example 1**:

    Input: nums = [5,6,7,8,9], x = 4
    Output: -1

**Example 2**:

    Input: nums = [3,2,20,1,1,3], x = 10
    Output: 5
    Explanation: The optimal solution is to remove the last three elements and the first two elements (5 operations in total) to reduce x to zero.

```python
"""
The number of elements removed equals n minus the number of elements that aren't removed.

Therefore, to find the minimum number of elements to remove, we can find the maximum number of elements to not remove!

So, instead of trying to find the minimum number of operations, why don't we focus on finding the longest subarray in the middle. One main thing to note is that our subarray should sum to sum - x (where sum is the sum of all elements in our array).
Why? because the middle elements are technically the ones we don't want. If the sum of the outer elements equals x, then we're looking for a middle sum of sum - x

"""
def minOperations(self, nums: List[int], x: int) -> int:
    # longest subarry (at middle) that sum to keep_sum
    keep_sum = sum(nums)-x
    l = 0
    cur_sum = 0
    max_window = -1

    for r in range(len(nums)):
        cur_sum+=nums[r]
        while cur_sum > keep_sum and l <= r:
            cur_sum -=nums[l]
            l+=1
        if cur_sum==keep_sum:
            max_window = max(max_window,r-l+1)

    return -1 if max_window==-1 else len(nums)-max_window
```