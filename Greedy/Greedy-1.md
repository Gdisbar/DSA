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