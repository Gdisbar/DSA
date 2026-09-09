## Pattern 1 — Interval Scheduling

| Section | Details |
| --- | --- |
| Recognition | Meetings • Activities • Schedule • Burst Balloons • Remove Overlapping |
| Core Idea | Always keep the interval that ends earliest — sort by end time first. |

****All 3 has time complexity $O(n \log n)$ - For sorting****

---

### 435. Non-overlapping Intervals

Given an array of intervals intervals where intervals[i] = [$start_{\text{i}}$, $end_{\text{i}}$], *return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping*.

Note that intervals which only touch at a point are non-overlapping. For example, [1, 2] and [2, 3] are non-overlapping.

**Example 1**:

    Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
    Output: 1
    Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

**Example 2**:

    Input: intervals = [[1,2],[1,2],[1,2]]
    Output: 2
    Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

**Example 3**:

    Input: intervals = [[1,2],[2,3]]
    Output: 0
    Explanation: You don't need to remove any of the intervals since they're already non-overlapping.

```python

"""
1. Minimum number of intervals to remove i.e remove fewest intervals
2. Which is nothing but maximum number of intervals we can should keep.
3. Then it comes under Maximum Meeting we can attend.

How to get that

- Again we sort by end times. Why? Because regardless of when a meeting starts, a meeting that ends first leaves more time for other meetings to take place. 

- We do not want a meeting that starts early and ends late, what we really care about is when the meeting ends and how much time it leaves for the other meetings. 

"""

def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[1])
    n = len(intervals)

    prev = 0
    count = 1

    for i in range(1, n):
        if intervals[i][0] >= intervals[prev][1]:
            prev = i
            count += 1

    return n - count

# Java

public int eraseOverlapIntervals(int[][] intervals) {
    int n = intervals.length;
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));

    int prev = 0;
    int count = 1;

    for (int i = 1; i < n; i++) {
        if (intervals[i][0] >= intervals[prev][1]) {
            prev = i;
            count++;
        }
    }
    return n - count;
}

```

### 452. Minimum Number of Arrows to Burst Balloons

There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array points where points[i] = [$x_{\text{start}}$, $x_{\text{end}}$] denotes a balloon whose horizontal diameter stretches between $x_{\text{start}}$ and $x_{\text{end}}$. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up directly vertically (in the positive y-direction) from different points along the x-axis. A balloon with $x_{\text{start}}$ and $x_{\text{end}}$ is burst by an arrow shot at ``x`` if $x_{\text{start}}$ <= ``x`` <= $x_{\text{end}}$. There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array points, *return the minimum number of arrows that must be shot to burst all balloons*.


**Example 1**:

    Input: points = [[10,16],[2,8],[1,6],[7,12]]
    Output: 2
    Explanation: The balloons can be burst by 2 arrows:
    - Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
    - Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].

**Example 2**:

    Input: points = [[1,2],[3,4],[5,6],[7,8]]
    Output: 4
    Explanation: One arrow needs to be shot for each balloon for a total of 4 arrows.

```python

"""
1. First, we sort the balloons based on their end coordinates. This allows us to iterate through the balloons in ascending order of their end coordinates.

2. If the start coordinate of the current balloon is **strictly greater** than the end coordinate of the previous balloon, it means these two balloons do not overlap, so we need to shoot another arrow. We increment the arrows count and update the prev variable to the end coordinate of the current balloon

3. After iterating through all balloons, arrows will contain the minimum number of arrows needed.

"""

# exactly same as 435 , only change is strictly greater 
#  points[i][0] > points[prev][1]

```

### 646. Maximum Length of Pair Chain

You are given an array of ``n`` pairs pairs where ``pairs[i]`` = [$left_{\text{i}}$, $right_{\text{i}}$] and $left_{\text{i}}$ < $right_{\text{i}}$.

A pair ``p2 = [c, d]`` follows a pair ``p1 = [a, b]`` if ``b < c``. A chain of pairs can be formed in this fashion.

*Return the length longest chain which can be formed*.

You do not need to use up all the given intervals. You can select pairs in any order.


**Example 1**:

    Input: pairs = [[1,2],[2,3],[3,4]]
    Output: 2
    Explanation: The longest chain is [1,2] -> [3,4].

**Example 2**:

    Input: pairs = [[1,2],[7,8],[4,5]]
    Output: 3
    Explanation: The longest chain is [1,2] -> [4,5] -> [7,8].
 
```python

```python
"""
So if we have 1 pair in out chain and nxt we have two options, say (1,10) & (1,3) which one should we pick next? 
Ofc the one with lower 2nd element to leave most room to pick next element for the chain. So this is what greedy says;
The dp approach is basically LIS(largest increasing subsequence)
"""

# exact same as 452 (the strictly greater version of 435)

```


## Pattern 2 — Interval Merging

| Section | Details |
|---|---|
| Recognition | Merge • Overlap • Combine |
| Core Idea | Sort by start, then merge while overlapping — Greedy + Sorting. |
| Representative Problems | 56, 57, 986 |

---

### 56. Merge Intervals

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.


**Example 1**:

    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

```python
"""
If it starts after current running maximum (a[0] > max_): There is a gap. The previous merged interval is complete, so save it and start a new running window.

If it overlaps or touches current range (a[0] <= max_): merge them by extending the end point (max_) if the new interval stretches further.
"""

def merge(self, arr: List[List[int]]) -> List[List[int]]:

    arr.sort(key = lambda x : x[0])
    merged_lst = [] 
    low = -10000
    high = -100000
    for i in range(len(arr)): 
        # Previous merge complete - start new interval
        if arr[i][0] > high: 
            if i != 0: 
                # Previous interval group is fully finished. 
                # append [low, high] to 
                merged_lst.append([low,high]) 
            # resets the window to start tracking the new interval
            high = arr[i][1] 
            low = arr[i][0] 
        # New interval overlaps with our current running window
        else: 
            # If this new interval extends further to the right
            # If arr[i][1] < high, it means the interval is completely 
            # swallowed inside the current window, so nothing changes
            if arr[i][1] >= high: 
                high = arr[i][1] 
    # Because the loop finishes without triggering the next 
    # "gap" condition for the very last interval, an extra check
    if high != -100000 and [low, high] not in merged_lst: 
        merged_lst.append([low, high]) 
    
    return merged_lst
```

### 57. Insert Interval

You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Two intervals are considered overlapping if they share at least one point.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.You can make a new array and return it.


**Example 1**:

    Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    Output: [[1,5],[6,9]]

```python

def insert(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    res = []
    n = len(intervals)
    start, end = newInterval[0], newInterval[1]
    i = 0
    
    # Add all intervals that come strictly before the new interval
    while i < n and intervals[i][1] < start:
        res.append(intervals[i])
        i += 1
        
    # Merge all overlapping intervals with the new interval
    while i < n and intervals[i][0] <= end:
        start = min(intervals[i][0], start)
        end = max(intervals[i][1], end)
        i += 1
        
    # Push the combined interval
    res.append([start, end])
    
    # Add all remaining intervals that come after
    while i < n:
        res.append(intervals[i])
        i += 1
        
    return res
```

## Pattern 3 — Jump Game

| Section | Details |
|---|---|
| Recognition | Jump • Reach • Maximum Distance |
| Core Idea | Track the farthest reachable position — optimization journey: Backtracking → DP → Greedy. |
| Representative Problems | 55, 45, 1306, 1345 |
| Interview Trap | Don't default to DP — track farthest reach instead. |

---

### 45. Jump Game II

You are given a 0-indexed array of integers nums of length n. You are initially positioned at index 0.

Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at index i, you can jump to any index (i + j) where:

    0 <= j <= nums[i] and
    i + j < n

Return the minimum number of jumps to reach index n - 1. The test cases are generated such that you can reach index n - 1.


**Example 1**:

    Input: nums = [2,3,1,1,4]
    Output: 2
    Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

```python
""" 
Linear Greedy 
-------

if jump_len == i:: This is the core greedy trigger. When our current traversal reaches the edge of our allowed jump boundary (jump_len), it means we are forced to make a jump to go further. We increment jump += 1 and update our boundary (jump_len) to the furthest point we've managed to scout (max_jump_len).

"""

def jump(self, nums: List[int]) -> int:
    n = len(nums)
    # In Jump Game I - we're returning True
    if n==1:
        return 0
    jump = 0
    jump_len = 0
    max_jump_len = 0
    
    for i in range(n-1):
        max_jump_len = max(max_jump_len,i+ nums[i])
        # In Jump Game I - we're returning False, as we reached end 
        # of our jump boundary but could couldn't reach the target 
        # assuming we can't make more jump (in case of Jump Game I)
        if jump_len==i: 
            jump+=1
            jump_len = max_jump_len
        # In Jump Game I - we're returning True, as we reached target
        if jump_len>=n-1:
            return jump
```

### 1306. Jump Game III

Given an array of non-negative integers arr, you are initially positioned at start index of the array. When you are at index i, you can jump to i + arr[i] or i - arr[i], check if you can reach any index with value 0.

Notice that you can not jump outside of the array at any time.

**Example 1**:

    Input: arr = [4,2,3,0,3,1,2], start = 5
    Output: true
    Explanation: 
    All possible ways to reach at index 3 with value 0 are: 
    index 5 -> index 4 -> index 1 -> index 3 
    index 5 -> index 6 -> index 4 -> index 1 -> index 3 

```python
"""
Because we can jump backwards (i - arr[i]), paths can loop infinitely (e.g., index 2 jumps to 5, and index 5 jumps back to 2). This turns the array into a directed graph, and the problem becomes checking if a path exists from start to a node with value 0.
"""
def canReach(self, arr: List[int], start: int) -> bool:
    visited = set()
    visited.add(start)
    for i in range(len(arr)):
        if len(visited)==0:
            break
        curr = visited.pop()
        if arr[curr]==0:
            return True
        # neighbours
        next_values = []
        if curr-arr[curr] >=0:
            next_values.append(curr-arr[curr])
        if curr+arr[curr] < len(arr):
            next_values.append(curr+arr[curr])
        
        for next_value in next_values:
            if next_value not in visited:
                visited.add(next_value)

    return False
```

### 1345. Jump Game IV

Given an array of integers arr, you are initially positioned at the first index of the array.

In one step you can jump from index i to index:

    i + 1 where: i + 1 < arr.length.
    i - 1 where: i - 1 >= 0.
    j where: arr[i] == arr[j] and i != j.

Return the minimum number of steps to reach the last index of the array.

Notice that you can not jump outside of the array at any time.

 
**Example 1**:

    Input: arr = [100,-23,-23,404,100,23,23,23,3,404]
    Output: 3
    Explanation: You need three jumps from index 0 --> 4 --> 3 --> 9. Note that index 9 is the last index of the array.


```python
"""
We start at index 0 and want to reach the last index (n - 1) in the minimum number of steps (shortest path). - classic Shortest Path in an Unweighted Graph problem - use BFS

The main catch is the third rule: if an array has millions of identical numbers, checking every single j every time we hit that number would take 
O(N^2) time and cause a Time Limit Exceeded (TLE).

"""
from collections import deque, defaultdict

def minJumps(self, arr: List[int]) -> int:
    n = len(arr)
    if n == 1:
        return 0
    val_to_indices = defaultdict(list)
    for idx,val in enumerate(arr):
        val_to_indices[val].append(idx)
    queue = deque([(0, 0)])  # (current_index, step_count)
    visited = set() # we can use - visited = {0}
    visited.add(0)
    while queue:
        curr, steps = queue.popleft()
        
        # If we reached the last index, return the steps
        if curr == n - 1:
            return steps
            
        # Possible next moves
        next_indices = []
        
        # Jump backward (if valid and not visited)
        if curr - 1 >= 0:
            next_indices.append(curr - 1)
            
        # Jump forward (if valid and not visited)
        if curr + 1 < n:
            next_indices.append(curr + 1)
            
        # Jump to any index with the exact same value (teleportation)
        if arr[curr] in val_to_indices:
            next_indices.extend(val_to_indices[arr[curr]])
            # Clear the list so we don't re-process these portals 
            # again (TLE prevention)
            del val_to_indices[arr[curr]]
            
        for next_idx in next_indices:
            if next_idx not in visited:
                visited.add(next_idx)
                queue.append((next_idx, steps + 1))
                
    return -1
```