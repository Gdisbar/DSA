//Approach/Observation
mid is on left : a[lo]<=a[mid]<=a[hi]
mid is on right : a[lo]>nums[hi]>=a[mid]
arrangement-1 : a[lo]<=a[mid]>=a[hi] //[4,5,6,|7|,0,1,2] , target=5,1 ,applicable for anything left of 0 i.e [4,5,6,7]
if(arr[lo]<=target && target<=a[mid])  
      hi=mid-1; //mid is on left,target a[lo...mid-1]
else lo=mid+1; //mid is on right,target is in a[mid+1...hi]
            
arrangement-2 : a[lo]>=a[mid]&&a[mid]<=a[hi] //[4,5,6,7,|0|,1,2] , target=5,1 applicable for [0,1,2]
if(a[mid]<=target&&target<=a[hi])
       lo=mid+1;
else hi=mid-1;
            
arrangement-3 : target<=a[mid] // mid is in [4,5,6,7] and target in [0,1,2] 
                    hi=mid-1
                else lo=mid+1

//Search in Rotated Sorted Array II

skip duplicate in left & right half //[4,5,6,6,7,0,1,2,4,4] --> [4,5,6,|7|,0,1,2] rest is exactly same

arrangement-1 : a[lo]<=a[mid] --> same as above //a[mid]>=a[hi] , loose bound might be omitted
arrangement-2 : same as above // arrangement-3 not needed covered under arrangement-2 + duplicate elemination

33. Search in Rotated Sorted Array
====================================
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown 
pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], 
nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become 
[4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, 
return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:

Input: nums = [1], target = 0
Output: -1


// 69 % faster,
int search(vector<int>& arr, int target) {
       int n=arr.size();
        int start=0,end=n-1;
        int mid=-1;
        while(start<=end){
            mid=start+(end-start)/2;
            if(arr[mid]==target) return mid;
            // there exists rotation; the middle element is in the left part of the array
            // If middle is on right part, we have: nums[left] > nums[right] >= nums[middle].
            // When middle is on right part and target is greater than nums[right] (means not in nums[middle .. right]), should search in left part.

            if(arr[start]<=arr[mid] && arr[mid]>=arr[end]){
                if(arr[start]<=target && target<=arr[mid])  
                    end=mid-1;
                else start=mid+1; // When (1) middle is on left part, (2) middle is on right part and target is in nums[middle ... right], continue search in right part.
            }
            // there exists rotation; the middle element is in the right part of the array
            // If middle is on left part, we have: nums[middle] >= nums[left] > nums[right]
            // When middle is on left part and target is smaller than nums[left] (means not in nums[left .. middle]), should search in right part.
            else if(arr[start]>=arr[mid] && arr[mid]<=arr[end]){
                if(arr[mid]<=target && target<=arr[end])
                        start=mid+1;
                else end=mid-1;
            }
             // there is no rotation; just like normal binary search
             // When (1) middle is on the right part, (2) middle is on left part and target is in nums[left .. middle], continue search in left part.
            else {
                if(target<=arr[mid])
                        end=mid-1;
                else start=mid+1;
            }
        }
        return -1;
     }




81. Search in Rotated Sorted Array II
=========================================
There is an integer array nums sorted in non-decreasing order (not necessarily 
with distinct values).Before being passed to your function, nums is rotated at 
an unknown pivot index k (0 <= k < nums.length) such that the resulting array is 
[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become 
[4,5,6,6,7,0,1,2,4,4].

Given the array nums after the rotation and an integer target, return true if 
target is in nums, or false if it is not in nums.

You must decrease the overall operation steps as much as possible.

 

Example 1:

Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

Example 2:

Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false



//25% faster, 96% less memory

bool search(vector<int>& nums, int target){
		int lo=0,hi=nums.size()-1;
        while(lo<=hi){
            while (lo < hi && nums[lo] == nums[lo+1]) ++lo; //To skip the duplicates number in start
            while (lo < hi && nums[hi] == nums[hi-1]) --hi; //To skip the duplicates number in end
            int mid=lo+(hi-lo)/2;
            if(nums[mid]==target) return true;
            
            if(nums[mid]>=nums[lo]){ //Finding which part is sorted
                if(target>=nums[lo] && target<=nums[mid]){
                    hi=mid-1;  //if within the sorted array we will use binary search in that
                }else{
                    lo=mid+1;  //if not then we will look on another sorted part.
                }
            }else{
                if(target>=nums[mid] && target<=nums[hi]){
                    lo=mid+1;
                }else{
                    hi=mid-1;
                }
            }
            
        }
        return false;
}


