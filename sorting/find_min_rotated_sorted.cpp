153. Find Minimum in Rotated Sorted Array
============================================
// Suppose an array of length n sorted in ascending order is rotated between 1 and n 
// times. For example, the array nums = [0,1,2,4,5,6,7] might become:

//     [4,5,6,7,0,1,2] if it was rotated 4 times.
//     [0,1,2,4,5,6,7] if it was rotated 7 times.

// Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in 
// the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

// Given the sorted rotated array nums of unique elements, return the minimum element 
// of this array.

// You must write an algorithm that runs in O(log n) time. //binary search

//starting element of part 2 is smallest--> [3,4,5,|1,2],[4,5,6,7,|0,1,2] , [inf|11,13,15,17]
// smallest element < element on it's left --> 1<5,7<0,11<inf , as part 1 is sorted
//last element of part 1 > element on its right --> 5>1,7>0,inf>11

// we need to approach starting element of part 2

// Example 1:

// Input: nums = [3,4,5,1,2]
// Output: 1
// Explanation: The original array was [1,2,3,4,5] rotated 3 times.

// Example 2:

// Input: nums = [4,5,6,7,0,1,2]
// Output: 0
// Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

// Example 3:

// Input: nums = [11,13,15,17]
// Output: 11
// Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 

// TC : 32% faster, 22% less memory

int findMin(vector<int>& nums) {
        int lo=0,hi=nums.size()-1;
        //2 3 4 5 1 --> case 1 ,a[lo]=2,a[hi]=1,a[mid]=4 --> a[lo]=5,a[hi]=1,a[mid]=5
        //5 1 2 3 4 --> a[lo]=5,a[hi]=4,a[mid]=2 --> a[lo]=5,a[hi]=1,a[mid]=5 ,mid-1 doesn't exist
        if(nums[lo]<=nums[hi]) return nums[lo]; // array not rotated
        while(lo<=hi){
            int mid=lo+(hi-lo)/2;
            // 3,4,5(mid),|1(mid+1),2 
            if(nums[mid]>nums[mid+1]){ //avoid overflow , last point of part 1
                return nums[mid+1];
            }
            // 3,4,5(mid-1),|1(mid),2 
            else if(nums[mid]<=nums[mid-1]){ //1st point of part 2
                return nums[mid];
            }
            //now normal binary search condition
            else if(nums[lo]<=nums[mid]) { //left part sorted,go to right
                lo=mid+1;
            }
            else if(nums[mid]<=nums[hi]) { //right part sorted,go to left
                hi=mid-1;
            }
        }
        return -1;
    }


// TC : 85% faster, 71% less memory

        //2 3 4 5 1 --> case 1 ,lo=5,hi=1,m=5
        //5 1 2 3 4 --> lo=5,hi=2 ,mid-1 doesn't exist
        //if(nums[lo]<=nums[hi]) return nums[lo]; // array not rotated

int findMin(vector<int>& nums) {
        int lo=0,hi=nums.size()-1;
        while(lo<hi){
            int mid=lo+(hi-lo)/2;
            if(nums[mid]>nums[hi]) lo=mid+1; //[4,5,6,7,|0,1,2] , minimum lies in right half
            else hi=mid;  //[11,|13,15,17] , minimum lies in left half
        }
        return nums[lo];
    }

154. Find Minimum in Rotated Sorted Array II
==============================================
// Suppose an array of length n sorted in ascending order is rotated between 1 and 
// n times. For example, the array nums = [0,1,4,4,5,6,7] might become:

//     [4,5,6,7,0,1,4] if it was rotated 4 times.
//     [0,1,4,4,5,6,7] if it was rotated 7 times.

// Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in 
// the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

// Given the sorted rotated array nums that may contain duplicates, return the 
// minimum element of this array.

// You must decrease the overall operation steps as much as possible.

 

// Example 1:

// Input: nums = [1,3,5]
// Output: 1

// Example 2:

// Input: nums = [2,2,2,0,1]
// Output: 0

// brute-force

int findMin(vector<int>& nums) {
        for (auto n: nums) if (n < nums[0]) return n;
        return nums[0];
    }

//30% faster, 19% less memory ---> 87% faster ,69% less memory


int findMin(vector<int>& nums) {
        int lo=0,hi=nums.size()-1;
        while(lo<hi){
            int mid=lo+(hi-lo)/2;
            if(nums[mid]>nums[hi]) { // min is inside right [mid+1,hi],left is sorted
                lo=mid+1;
            }
            else if(nums[mid]<nums[lo]){ //min is inside left [lo+1,mid],right is sorted
                hi=mid;
                lo++;  
            }
            else { // nums[lo] <= nums[mid] <= nums[hi] ,min is nums[lo]
            	hi--;
            }
            //nums[lo] > nums[mid] > nums[hi], impossible
        }
        return nums[lo]; // here lo=hi
    }

// 18% faster , 19% less memory , recursive binary search

class Solution {
private:
    int search(vector<int> &nums,int lo,int hi){
        if(lo==hi) return nums[lo];
        int mid=lo+(hi-lo)/2;
        //min lies on right,left sorted
        if(nums[mid]>nums[hi]) return search(nums,mid+1,hi); 
        //min lies on left,right sorted
        if(nums[mid]<nums[hi]) return search(nums,lo,mid); 
        return search(nums,lo,hi-1);
    }
public:
    int findMin(vector<int>& nums) {
        return search(nums,0,nums.size()-1);
    }
};