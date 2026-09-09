
# HASHING DECISION TREE

```text
Hashing
│
├── Need fast lookup / duplicate check? → HashSet
│
├── Need value → index? → HashMap
│
├── Need counting? → Frequency Map
│
├── Need complement (pair sum)? → HashMap
│
├── Need grouping by signature? → HashMap<List>
│
├── Need ordering? → TreeMap / Ordered structure
│
├── Need sliding frequencies? → HashMap + Sliding Window
│
├── Need prefix information? → HashMap + Prefix Sum
│
└── Need O(1) insert/delete + ordering/eviction? → HashMap + Linked List
```

Whenever you write `for i { for j { ... } }`, ask: *can I remember information from previous iterations?* If yes, HashMap is probably the optimization.

---

# CLASSIFY INTO PATTERN

| Pattern | When to Recognize | Common Problems |
|---|---|---|
| 1. Existence Lookup | Exists? Seen before? Duplicate? | Contains Duplicate, Happy Number, Longest Consecutive Sequence |
| 2. Value → Index | Find pair, need indices, complement | Two Sum |
| 3. Frequency Counting | Most frequent, majority, anagram | Top K Frequent Elements, Majority Element |
| 4. Complement Search | Pair/triplet/quadruplet sums to target | Two Sum, 4Sum |
| 5. Grouping | Cluster by signature/key | Group Anagrams |
| 6. Prefix + HashMap | Subarray sum = K, running sum, exactly K | Subarray Sum Equals K, Continuous Subarray Sum |
| 7. HashMap + Sliding Window | Longest/shortest substring with a frequency constraint | Longest Substring Without Repeating Characters, Minimum Window Substring |
| 8. HashMap + Sorting | Sort by frequency/rank | Sort Characters By Frequency |
| 9. HashMap + Heap | Top K, streaming, median | Top K Frequent Elements, Kth Largest Element in a Stream |
| 10. Hashing States | Visited configuration, board state | Valid Sudoku |
| 11. Hashing Signatures | Isomorphic, canonical form | Isomorphic Strings, Group Anagrams |
| 12. Custom Hashing | Complex/compound keys (pairs, tuples, coordinates) | Max Points on a Line |
| 13. HashMap + Linked List (O(1) Design) | Need O(1) with eviction/order/random access | LRU Cache, Insert Delete GetRandom O(1) |

---

## Pattern 1 — Existence Lookup

| Section | Details |
|---|---|
| Recognition | Exists? • Already Seen? • Contains? • Duplicate? • Visited? |
| Core Idea | Brute-force search is O(n²) per lookup; a HashSet makes membership O(1). |
| Time / Space | O(n) / O(n) |
| Common Trap | People reach for HashMap when a HashSet is enough. |

**Variations**

| Variation | Core Idea | Representative Problems |
|---|---|---|
| Duplicate Detection | Seen-before check | 217, 219, 220 |
| Missing Value | Complement of a HashSet scan | 268, 645, 448 |
| Happy Number | Cycle detection on a number sequence | 202 |
| Cycle Detection by State | Repeated-state detection | 874 |
| Consecutive Sequence | Only start counting a run if `n-1` is **not** in the set — every run gets counted exactly once, from its smallest member | 128 |

---

### 202. Happy Number

Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.

**Example 1**:

    Input: n = 19
    Output: true
    Explanation:
    1**2 + 9**2 = 82
    8**2 + 2**2 = 68
    6**2 + 8**2 = 100
    1**2 + 0**2 + 0**2 = 1

```python
"""
If the number is NOT happy (it has a cycle): fast and slow will eventually land on the exact same number inside that cycle. When slow == fast, the loop breaks. Since that intersection value is not 1, slow == 1 evaluates to false.

If the number IS happy (it reaches 1 - Once it hits 1, it stays 1 forever (1**2 = 1). This is a Happy Number.): Both pointers will eventually get pulled into the sinkhole of 1. Because numsum(1) == 1, both slow and fast will stick at 1, meaning they equal each other at 1, the loop breaks, and slow == 1 evaluates to true.
"""
def isHappy(self, n: int) -> bool:
    slow = n; 
    fast = n; 
    while(True):  
        slow = self.numsum(slow);  
        fast = self.numsum(self.numsum(fast)); 
        if(slow != fast): 
            continue
        else: 
            break
    return (slow == 1)

def numsum(self,n : int) -> int:
    sums = 0
    while n != 0:
        sums += (n%10)**2
        n = int(n/10)
    return sums
    
```

### 128. Longest Consecutive Sequence

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

**Example 1**:

    Input: nums = [0,3,7,2,5,8,4,6,0,1]
    Output: 9

```python
"""
Store all numbers in an unordered_set to remove duplicates.
For every number val:
    1. If val - 1 exists → it's not the start, so skip it.
    2. Otherwise, start counting the consecutive sequence 
        i.e count of val+1 that is in the set
Keep updating the maximum length found after loop i.e val+1 not in set.
"""

mx_len = 0
for n in set1:
    if n-1 in set1:
        continue
    val = n
    cur = 1
    while val+1 in set1:
        val+=1
        cur+=1
    mx_len= max(mx_len,cur)

```

### 219. Contains Duplicate II

Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

**Example 1**:

    Input: nums = [1,2,3,1], k = 3
    Output: true

```python


def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    st = set()
    i = 0
    for j in range(len(nums)):
        # outside window
        if abs(j-i)>k:
            st.discard(nums[i])
            i+=1
        # have seen in past
        if nums[j] in st:
            return True
        st.add(nums[j])

    return False
        
```

### 220. Contains Duplicate III

You are given an integer array nums and two integers indexDiff and valueDiff.

Find a pair of indices (i, j) such that:

    i != j,
    abs(i - j) <= indexDiff.
    abs(nums[i] - nums[j]) <= valueDiff, and
    Return true if such pair exists or false otherwise.


**Example 1**:

    Input: nums = [1,2,3,1], indexDiff = 3, valueDiff = 0
    Output: true
    Explanation: We can choose (i, j) = (0, 3).
    We satisfy the three conditions:
    i != j --> 0 != 3
    abs(i - j) <= indexDiff --> abs(0 - 3) <= 3
    abs(nums[i] - nums[j]) <= valueDiff --> abs(1 - 1) <= 0

**Example 2**:

    Input: nums = [1,5,9,1,5,9], indexDiff = 2, valueDiff = 3
    Output: false
    Explanation: After trying all the possible pairs (i, j), we cannot satisfy the three conditions, so we return false.

```python
# In 219, a duplicate means exact equality (nums[i] == nums[j]).

# In 220, a duplicate means approximate equality (abs(nums[i] - nums[j]) <= t).

# Because a standard set can only check for exact matches, 220 uses buckets to bridge the gap.

# In 219 solution, you maintain a sliding window of size k using a set. When the window exceeds size k, we drop the oldest element (nums[i]) and slide left pointer forward.
# 
# The 220 bucket solution does the exact same sliding window management, but instead of checking raw values in a set, it groups numbers into buckets of size t + 1:
# 
# The Exact-Match Equivalent (Same Bucket):
# If two numbers fall into the same bucket, their difference is automatically less than or equal to t. Checking if bucket in buckets: is the 220 equivalent of if nums[j] in st: check in 219.
# 
# The Near-Match Check (Adjacent Buckets):Because numbers in neighboring buckets can also be within distance t, the code checks bucket - 1 and bucket + 1, verifying their actual difference (<= t).
# 
# Sliding Window Clean-up (Maintaining Size $k$):In 219 code, we shrink the window with st.discard(nums[i]) when abs(j - i) > k.
# In the 220 code, because the loop uses i as the current index (acting like j), the element falling out of the window from the left is nums[i - k]. When len(buckets) > k, it finds the bucket of that old element and deletes it:

"""

[1,5,2,4,3,9,1,5,9], k = 2, t = 3

perform - n/(t+1)

1 // (3+1) = 0
5 // (3+1) = 1
2 // (3+1) = 0
4 // (3+1) = 1
3 // (3+1) = 0
9 // (3+1) = 2

Here, Bucket[0] will contain numbers 0,1,2,3.
Bucket[1] will contain numbers 4,5,6,7.
Bucket[2] will contain numbers 8,9,10,11.

On observing carefully, we can see that the absolute difference
between any two numbers in any bucket is at most t, which is what we want.

Also, there can be a case where the neighbouring bucket has some number
whose absolute difference with a number in the current bucket is at most t.
For instance, 2 lies in Bucket[0] and 4 lies in Bucket[1] and 4 - 2 = 2 < 3 (=t).
This can only happen in neighbouring buckets. Therefore, we need to check for this too.

"""

def containsNearbyAlmostDuplicate(self, nums: list[int], k: int, t: int) -> bool:
    n = len(nums)
    
    if n == 0 or k < 0 or t < 0:
        return False
    # hashmap - bucketID : num[i]_belonging_to_this_bucket   
    buckets = {}
    bucket_size = t + 1
    
    for i in range(n):
        # assign to proper bucket
        bucket = nums[i] // bucket_size
        
        # Checking if bucket in buckets: is the 220 equivalent 
        # of if nums[j] in st: check in 219.

        if bucket in buckets:
            return True
        # Because numbers in neighboring buckets can also be within 
        # distance t, the code checks bucket - 1 and bucket + 1, 
        # verifying their actual difference (<= t).
        if (bucket - 1) in buckets and (nums[i] - buckets[bucket - 1]) <= t:
            return True
        if (bucket + 1) in buckets and (buckets[bucket + 1] - nums[i]) <= t:
            return True
            
        buckets[bucket] = nums[i]
        
        # Maintain the sliding window of size k using pure index
        if len(buckets) > k:
            key_to_remove = nums[i - k] // bucket_size
            del buckets[key_to_remove]
            
    return False
```


## Pattern 2 — Value → Index

| Section | Details |
|---|---|
| Recognition | Find pair • Need indices • Earlier occurrence • Complement |
| Core Idea | Store `value → index` (or `value → count`) while scanning once; look up the complement instantly instead of nested loops. |
| Time / Space | O(n) / O(n) |
| Representative Problems | 1, 167, 170 |
| Common Trap | Need the *original* indices? Don't sort first. |

---
