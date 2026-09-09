
# DECISION TREE

```text
│
├── Need largest/smallest, or Top K? → Heap
├── Need running answer as data streams in? → Heap
├── Need median after every insertion? → Two Heaps
├── Need earliest-finishing resource? → Min Heap
├── Need merge many sorted sequences? → Min Heap
├── Need weighted shortest path? → Graph + Min Heap
└── Need everything sorted? → Don't use Heap
```

---

# CLASSIFY INTO PATTERN

| Pattern | When to Recognize | Common Problems |
|---|---|---|
| 1. Top K | Largest/smallest K elements | Kth Largest Element in an Array, K Closest Points to Origin |
| 2. Kth Element | Single order statistic, not a list | Kth Largest Element in an Array |
| 3. Merge K Sorted Structures | Merge many sorted lists/streams | Merge k Sorted Lists |
| 4. Stream Processing | Online/incoming data | Kth Largest Element in a Stream |
| 5. Two Heaps | Running median | Find Median from Data Stream |
| 6. Scheduling | Tasks, CPU, meeting rooms | Task Scheduler |
| 7. Greedy + Heap | Always pick the best available | IPO, Furthest Building You Can Reach |
| 8. Graph + Heap | Weighted shortest path | Network Delay Time (Dijkstra) |
| 9. Interval + Heap | Meeting rooms, overlaps | Meeting Rooms II |
| 10. Frequency + Heap | Top/most frequent | Top K Frequent Elements |
| 11. Heap for DP Optimization | Score/window DP, often superseded by Monotonic Queue | Maximum Sliding Window (as DP) |
| 12. Multiple Heaps | Available vs. busy resources | Meeting Rooms III |

---

## Building a Heap: O(n), Not O(n log n)

Inserting n elements one at a time is n × O(log n) = O(n log n). But **heapify** — sifting down from the last non-leaf node upward on an existing array — is O(n), since most nodes sit near the bottom where sift-down does almost no work. Python's `heapq.heapify`, C++'s `std::make_heap`, and similar library calls already use this. Claiming O(n log n) to build a heap is a common self-inflicted interview stumble.

---

## Pattern 1 — Top K

| Section | Details |
|---|---|
| Recognition | Top K • Largest/Smallest K • K Frequent • K Closest |
| Core Idea | Sort is O(n log n); since you only need K, a heap gets you O(n log k). |
| Representative Problems | 658, 973, 1985 |
| Common Rule | Need Largest K → **Min** Heap. Need Smallest K → **Max** Heap. Counterintuitive — understand why. |

215 (Kth Largest), 347 & 692 (Top K Frequent), and 373 (K Pairs with Smallest Sums) are Top-K problems too, but each needs one more specific technique — covered under Kth Element, Frequency + Heap, and Merge K Sorted Structures respectively.

**Implementation Technique — Fixed-Size Heap:** push, then pop whenever size > K. Heap never exceeds O(k) space. Don't sort everything and truncate.

---

### 973. K Closest Points to Origin

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).


**Example 1**:

        Input: points = [[1,3],[-2,2]], k = 1
        Output: [[-2,2]]
        Explanation:
        The distance between (1, 3) and the origin is sqrt(10).
        The distance between (-2, 2) and the origin is sqrt(8).
        Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
        We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].

```python
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    def calculate_distance(p):
        return math.sqrt(p[0]**2+p[1]**2)
    # Need to remove larger distance from top of heap - max_heap
    data = []
    for p in points:
        dist_origin = calculate_distance(p)
        heapq.heappush(data,(-1*dist_origin,p))
        if len(data)>k:
            heapq.heappop(data)
    return [r for _,r in data]
```

## Pattern 2 — Kth Element

| Section | Details |
|---|---|
| Recognition | Kth Largest/Smallest • Median • Order Statistic |
| Core Idea | "Find just the Kth value" is a more precise ask than "return the Top K list." |
| Representative Problems | 215, 378, 786 |
| Optimization | Sort → Heap → QuickSelect → Median of Medians (Awareness) |

---

### 786. K-th Smallest Prime Fraction

You are given a sorted integer array arr containing 1 and prime numbers, where all the integers of arr are unique. You are also given an integer k.

For every i and j where 0 <= i < j < arr.length, we consider the fraction arr[i] / arr[j].

Return the kth smallest fraction considered. Return your answer as an array of integers of size 2, where answer[0] == arr[i] and answer[1] == arr[j].


**Example 1**:

    Input: arr = [1,2,3,5], k = 3
    Output: [2,5]
    Explanation: The fractions to be considered in sorted order are:
    1/5, 1/3, 2/5, 1/2, 3/5, and 2/3.
    The third fraction is 2/5.

**Example 2**:

    Input: arr = [1,7], k = 1
    Output: [1,7]
 

Constraints:
    2 <= arr.length <= 1000

```python
"""TLE even at given constraints : O(n**2)"""
def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
    data = []
    for i in range(len(arr)-1):
        for j in range(i+1,len(arr)):
            heapq.heappush(data,((arr[i]/arr[j]),[arr[i],arr[j]]))
    # Pop k times to reach the k-th smallest fraction
    ans = []
    for _ in range(k):
        ans = heapq.heappop(data)[1]


""" using smaller heap - O(N)"""

def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
    n = len(arr)
    heap = []

    # Initialize heap with 1/arr[i] which are the smallest,i=0
    for j in range(1, n):
        heapq.heappush(heap, (arr[0] / arr[j], 0, j))
    
    # Pop k-1 times, pushing the next valid fraction for that denominator
    for _ in range(k - 1):
        val, i, j = heapq.heappop(heap)
        if i + 1 < j:
            heapq.heappush(heap, (arr[i + 1] / arr[j], i + 1, j))
            
    # The k-th smallest fraction is now at the root of the heap
    _, i, j = heap[0]
    return [arr[i], arr[j]]

"""Optimized Binary Search"""

def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
    n = len(arr)
    left, right = 0.0, 1.0
    
    while left < right:
        mid = (left + right) / 2
        count = 0
        p, q = 0, 1
        j = 1
        
        # Count how many fractions arr[i] / arr[j] <= mid
        for i in range(n - 1):
            while j < n and arr[i] >= mid * arr[j]:
                j += 1
            
            # All elements from j to n-1 form a fraction <= mid with arr[i]
            count += (n - j)
            
            # Track the maximum fraction that is still <= mid
            if j < n and arr[i] * q > p * arr[j]:
                p, q = arr[i], arr[j]
        
        if count == k:
            return [p, q]
        elif count < k:
            left = mid
        else:
            right = mid
    
    return []
```