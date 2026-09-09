
# ARRAY DECISION TREE

```text
Array
│
├── Need range information? → Prefix Sum
│
├── Need contiguous answer?
│      ├── Fixed size? → Fixed Window
│      └── Variable size? → Variable Window
│
├── Need pair/triplet? → Sort → Two Pointer
│
├── Need Top K? → Heap
│
├── Need frequency? → HashMap
│
├── Need in-place rearrangement?
│      ├── Swap Strategy
│      ├── Cycle Sort
│      └── Partition
│
├── Need repeated queries?
│      ├── Prefix/Suffix
│      └── Sparse Table
│
└── Need optimization over a value range? → Binary Search on Answer
```

---

# CLASSIFY INTO PATTERN

| Pattern | When to Recognize | Common Problems |
|---|---|---|
| 1. Linear Scan / Simulation | Single pass, running state, never need to look back | Best Time to Buy/Sell Stock, Maximum Subarray, Majority Element |
| 2. Prefix Computation | Range sum/count queries, repeated subarray questions | Range Sum Query, Subarray Sum Equals K, Continuous Subarray Sum |
| 3. Prefix + Suffix | Need info from both sides of every index at once | Product of Array Except Self, Trapping Rain Water, Candy |
| 4. Partitioning | Rearrange around a condition/pivot | Sort Colors, Move Zeroes |
| 5. Cyclic Placement | Values confined to 1..n, find missing/duplicate | Missing Number, Find the Duplicate Number, First Missing Positive |
| 6. Matrix Simulation | 2D grid — rotate, traverse, spiral, diagonal | Rotate Image, Spiral Matrix, Diagonal Traverse |
| 7. Rearrangement | Reorder elements in place | Next Permutation, Merge Sorted Array, Rotate Array |
| 8. Sorting Enables Structure | Sorting reveals pairing/interval structure | Merge Intervals, 3Sum, Non-overlapping Intervals |
| 9. Range Query Thinking | Many repeated queries over ranges, some with updates | Range Sum Query - Mutable |
| 10. Binary Search on Array Property | Minimum/maximum feasible value, not a target index | Koko Eating Bananas, Capacity To Ship Packages Within D Days |

Notice how Sliding Window isn't here — it's covered in its own chapter, since it's an evolution of these ideas rather than a peer.


---

## Pattern 1 — Linear Scan / Simulation

| Section | Details |
|---|---|
| Recognition | Traverse once • Maintain state • Running max/min • Track previous value • Process events • Transform array |
| Core Idea | Only the previous state matters — carry it forward in a single variable instead of rescanning. O(n²) → O(n). |
| Time / Space | O(n) / O(1) |
| Common Trap | People reach for a HashMap by reflex. Often one running variable is enough. |

**Variations**

| Variation | Core Idea | Representative Problems |
|---|---|---|
| Running Maximum / Minimum | Track the best value seen so far | 121, 918 |
| Kadane's Algorithm | `cur = max(nums[i], cur+nums[i])`; drop everything before the moment `cur` falls below the element itself — this **is** a 1D DP, not "just a variable" | 53 |
| Running Product | Track running max **and** min product — negatives flip which is which | 152 |
| Running Balance | Track current balance / score / altitude | 1732 |
| Boyer-Moore Voting | Candidate + counter; counter hits 0 → swap candidate. O(1)-space alternative to HashMap counting | 169 |
| State Machine Simulation | States: Buy / Sell / Cooldown / Holding | 122, 309, 714 |
| Event Processing | Arrival / Departure / Booking — elaborated fully as Difference Array & Sweep Line in Pattern 2 below | *(see Pattern 2)* |

---

### 121. Best Time to Buy and Sell Stock

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

**Example 1**:

    Input: prices = [7,1,5,3,6,4]
    Output: 5
    Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
    Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

```python
# Maximizing the subarray sum of these differences is mathematically
#  equivalent to finding the maximum profit between a buy price 
# and a subsequent sell price.
curr_max += prices[i] - prices[i-1]
# Kadane's reset step - the accumulated profit up to yesterday 
# drops below zero, carrying it forward is counterproductive, 
# so the algorithm discards it and starts a fresh subarray
if curr_max<0:
    curr_max = 0
# global maximum profit encountered across all possible trading intervals
max_so_far = max(max_so_far,curr_max)

```

### 122. Best Time to Buy and Sell Stock II

You are given an integer array prices where prices[i] is the price of a given stock on the i-th day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can sell and buy the stock multiple times on the same day, ensuring you never hold more than one share of the stock.

Find and return the maximum profit you can achieve.

**Example 1**:

    Input: prices = [7,1,5,3,6,4]
    Output: 7
    Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
    Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
    Total profit is 4 + 3 = 7.

```python
"""
This variant accumulates every positive daily price difference rather than resetting when the running total drops below zero. This allows you to buy and sell multiple times (as many transactions as you want, provided you sell before buying again), whereas the previous snippet was for a single transaction.
"""
curr_max += prices[i] - prices[i-1]
# Whenever the daily change is positive, you instantly lock in 
# the profit by pretending to buy yesterday and sell today.
if curr_max>0:
    ans += curr_max
# Instantly resets the tracking container so you are ready to 
# evaluate the next day's movement as a potential new transaction
curr_max = 0

```

### 53. Maximum Subarray

Given an integer array nums, find the subarray with the largest sum, and return its sum.

**Example 1**:

    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.

```python
"""Classic Kadane"""
if len(nums)==1:
    return nums[0]

mx_sum = nums[0],cur_sum =  nums[0]
for i in range(1,len(nums)):
    # handles negative numbers natively without needing 
    # an explicit zero-tracking or prefix-min offset - below
    cur_sum=max(nums[i]+cur_sum,nums[i])
    mx_sum = max(cur_sum,mx_sum)

# or 

for i in range(len(nums)):
    cur_sum += nums[i];
    # Max subarray ending here is current 
    # prefix minus the lowest past prefix
    max_sum= max(max_sum, cur_sum - min_prefix_sum);
    min_prefix_sum = min(min_prefix_sum, cur_sum);
```

## Pattern 2 — Prefix Computation

| Section | Details |
|---|---|
| Recognition | Range Sum • Running Sum • Subarray • Repeated Queries • Exactly K • Cumulative |
| Core Idea | Brute force recomputes every range in O(n²). Since the left part always repeats, store cumulative information once and answer instantly. |
| Time / Space | O(n) build, O(1) per query / O(n) |

**Variations**

| Variation | Core Idea | Representative Problems |
|---|---|---|
| Prefix Sum | Cumulative sum array | 303, 724, 560, 930, 974, 1074 |
| Prefix XOR | Parity / odd-even tracking via XOR | 1310, 1442, 2425 |
| Prefix Product | Cumulative product | 1352 |
| Prefix Frequency | Cumulative counts | 525, 1124 |
| Prefix Min / Max | 121, 152, and 42 are stronger fits under Pattern 1 (Running Max) and Pattern 3 (Prefix+Suffix) — not repeated here | 2012 |
| Difference Array *(Advanced)* | Range updates in O(1); build once, take prefix sum at the end | 370, 1094, 1109 |
| Sweep Line *(Advanced)* | Events/intervals sorted by coordinate | 56, 57, 253, 759 |

| Optimization Journey |
|---|
| Calculate every range → Store cumulative → Need fast lookup → Prefix Array → Need counting → Prefix + HashMap → Need updates → Difference Array → Need dynamic updates → Fenwick Tree |

---