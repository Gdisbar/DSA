# BINARY SEARCH (Search & Sort)

**Core Idea**
Binary Search is not a searching algorithm — it's an **optimization framework** for any monotonic answer space. Ask: *"If X works, will every larger (or smaller) value also work?"* If yes, Binary Search applies — even when there's no array in sight.

---

# DECISION TREE

```text
│
├── Need exact value? → Classic Binary Search
├── Need first/last occurrence? → Lower/Upper Bound
├── Need min/max feasible answer? → Binary Search on Answer
├── Need rotated array? → Modified Binary Search
├── Need peak? → Binary Search on Slope
├── Need partition across two sorted arrays? → Binary Search on Partition
└── Need continuous value? → Binary Search on Real Numbers (Awareness)
```


---

## Pattern 1 — Lower / Upper Bound

| Section | Details |
|---|---|
| Recognition | First • Last • Insert Position • Occurrence • Boundary |
| Core Idea | Find the first TRUE or last FALSE in a monotonic `F F F T T T` sequence. |


### 34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.



**Example 1**:
    Input: nums = [5,7,7,8,8,10], target = 8
    Output: [3,4]

**Example 2**:

    Input: nums = [5,7,7,8,8,10], target = 6
    Output: [-1,-1]


```python

def lowerBound(self, nums: List[int], target: int) -> int:
    l,r = 0,len(nums)-1
    result = -1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]>=target:
            result=mid
            r = mid-1 # look in left for smaller
        else: 
            l = mid+1

    return result

def upperBound(self, nums: List[int], target: int) -> int:
    l,r = 0,len(nums)-1
    result = -1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]<=target:
            result=mid
            l = mid+1 # look in right for larger
        else:
            r = mid-1
            
    return result

def binarySearch(self, nums: List[int], target: int) -> int:
    result = -1
    l, r = 0,len(nums)-1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]==target:
            return mid
        elif nums[mid] > target:
            r = mid -1 # look for smaller in left
        else:
            l = mid + 1
    return result


def searchRange(self, nums: List[int], target: int) -> List[int]:
    if self.binarySearch(nums,target)!=-1:
        lower_bound = self.lowerBound(nums,target)
        upper_bound = self.upperBound(nums,target)
        return [lower_bound,upper_bound]
    else:
        return [-1,-1]
```

```java
import java.util.List;

class Solution {
    public int lowerBound(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        int result = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] >= target) {
                result = mid;
                r = mid - 1; // look in left for smaller
            } else {
                l = mid + 1;
            }
        }
        return result;
    }

    public int upperBound(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        int result = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= target) {
                result = mid;
                l = mid + 1; // look in right for larger
            } else {
                r = mid - 1;
            }
        }
        return result;
    }

    public int binarySearch(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                r = mid - 1; // look for smaller in left
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        if (binarySearch(nums, target) != -1) {
            int lowerBound = lowerBound(nums, target);
            int upperBound = upperBound(nums, target);
            return new int[]{lowerBound, upperBound};
        } else {
            return new int[]{-1, -1};
        }
    }
}
```

**35 (Search Insert Position) belongs under Pattern 2 (Lower/Upper Bound) — "return where it would be inserted" is literally a boundary search. Here it's Lower Bound**

---

## Pattern 2 — Search in Rotated Array

| Section | Details |
|---|---|
| Recognition | Rotated • Shifted • Pivot • Circular |
| Core Idea | At least one half is always sorted — use that half to decide direction. |
| Common Trap | Duplicates change the pattern entirely. |

---

### 33. Search in Rotated Sorted Array

There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly left rotated at an unknown index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be left rotated by 3 indices and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

**Example 1**:
    Input: nums = [4,5,6,7,0,1,2], target = 0
    Output: 4

**Example 2**:
    Input: nums = [4,5,6,7,0,1,2], target = 3
    Output: -1

```python
def findPivot(self,nums:List[int]) -> int:
    l , r = 0, len(nums)-1
    while l < r:
        # # duplicate check
        # while l < r and nums[l]==nums[l+1]:
        #     l+=1
        # while r < l and nums[r]==nums[r-1]:
        #     r-=1
        mid = l + (r-l)//2
        if nums[mid]<= nums[r]:
            r = mid
        else:
            l = mid+1
    return l # return pivot index

def binarySearch(self,nums: List[int],l:int,r:int,target:int):
    while l<=r:
        m = l + (r-l)//2
        if nums[m]==target:
            return m
        elif nums[m] > target:
            r-=1
        else:
            l+=1
    return -1

def search(self, nums: List[int], target: int) -> int:
    n = len(nums)
    pivot = self.findPivot(nums)
    if nums[pivot]==target:
        return pivot

    idx = self.binarySearch(nums,0,pivot-1,target)
    if idx !=-1:
        return idx
    idx = self.binarySearch(nums,pivot+1,n-1,target)
    return idx
```

### 81. Search in Rotated Sorted Array II

There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.

**Example 1**:
    Input: nums = [2,5,6,0,0,1,2], target = 0
    Output: true

**Example 2**:
    Input: nums = [2,5,6,0,0,1,2], target = 3
    Output: false

**same as above just comment out the ducplicate checking**

### 153. Find Minimum in Rotated Sorted Array

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.


**Example 1**:
    Input: nums = [3,4,5,1,2]
    Output: 1
    Explanation: The original array was [1,2,3,4,5] rotated 3 times.

**Example 2**:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

**value at pivot index is the smallest value - ```self.findPivot(nums)```**


## Pattern 4 — Peak Finding

| Section | Details |
|---|---|
| Recognition | Peak • Mountain • Slope |
| Core Idea | Compare `mid` with `mid+1`; move toward the uphill side. |
| Representative Problems | 162, 852, 1095, 1901 |

852 resembles a rotated-array search too, but its defining structure is the mountain shape — that's why it lives here.

---
