628. Maximum Product of Three Numbers
========================================
Given an integer array nums, find three numbers whose product is maximum 
and return the maximum product.

 

Example 1:

Input: nums = [1,2,3] 
Output: 6

--> mx1 : 3 mx2 : 2 mx3 : 1 mn1 : 1 mn2 : 2

Example 2:

Input: nums = [1,2,3,4]
Output: 24

--> mx1 : 4 mx2 : 3 mx3 : 2 mn1 : 1 mn2 : 2

Example 3:

Input: nums = [-1,-2,-3]
Output: -6

--> mx1 : -1 mx2 : -2 mx3 : -3 mn1 : -3 mn2 : -2

int maximumProduct(vector<int>& nums) {
        int mn1=INT_MAX,mn2=INT_MAX,mx1=INT_MIN,mx2=INT_MIN,mx3=INT_MIN;
        for(int i = 0;i<nums.size();i++){
            if(nums[i]>mx1){
                mx3=mx2;
                mx2=mx1;
                mx1=nums[i];
            }
            else if(nums[i]>mx2){
                mx3=mx2;
                mx2=nums[i];
            }
            else if(nums[i]>mx3){
                mx3=nums[i];
            }
            if(nums[i]<mn1){
                mn2=mn1;
                mn1=nums[i];
            }
            else if(nums[i]<mn2){
                mn2=nums[i];
            }
        }
       //cout<<"mx1 : "<<mx1<<" mx2 : "<<mx2<<" mx3 : "<<mx3<<" mn1 : "<<mn1<<" mn2 : "<<mn2<<endl;

        return max(mx1*mn1*mn2,mx1*mx2*mx3);
        
    }


747. Largest Number At Least Twice of Others
================================================
// You are given an integer array nums where the largest integer is unique.

// Determine whether the largest element in the array is at least twice as much as 
// every other number in the array. If it is, return the index of the largest element, 
// or return -1 otherwise.

 

// Example 1:

// Input: nums = [3,6,1,0]
// Output: 1
// Explanation: 6 is the largest integer.
// For every other number in the array x, 6 is at least twice as big as x.
// The index of value 6 is 1, so we return 1.

// Example 2:

// Input: nums = [1,2,3,4]
// Output: -1
// Explanation: 4 is less than twice the value of 3, so we return -1.

// Example 3:

// Input: nums = [1]
// Output: 0
// Explanation: 1 is trivially at least twice the value as any other number because 
// there are no other numbers.

int dominantIndex(vector<int>& nums) {
        if(nums.size()==1&&nums[0]==1) return 0;
        //if(nums.size()==1&&nums[0]==1) return 0;
        int mx1=INT_MIN,mx2=INT_MIN;
        for(int i = 0;i<nums.size();i++){    
            if(nums[i]>mx1){
                mx2=mx1;
                mx1=nums[i];
            }
            else if(nums[i]>mx2){
                mx2=nums[i];
            }
        
        }
        if(mx1>=2*mx2) return find(nums.begin(),nums.end(),mx1)-nums.begin();
        else return -1;
    }

53. Maximum Subarray //Kadane 
================================
// Given an integer array nums, find the contiguous subarray (containing at 
// least one number) which has the largest sum and return its sum.

// A subarray is a contiguous part of an array.

 

// Example 1:

// Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
// Output: 6
// Explanation: [4,-1,2,1] has the largest sum = 6.

//fastest greedy version
int maxSubArray(vector<int>& nums) {
        int mx1=INT_MIN,mx2=0;
        for(int i=0;i<nums.size();i++){
            mx2+=nums[i];
            if(mx2>mx1){
                mx1=mx2;
            }
            if(mx2<0){
                mx2=0;
            }
        }
        return mx1;
    }

// dp version of kadane is slower

int maxSubArray(vector<int>& nums) {
        int sum = nums[0];
        int result = sum;
        for (int i = 1; i < nums.size(); ++i) {
            if (sum > 0)
                sum += nums[i];
            else
                sum = nums[i];
            result = max(result, sum);
        }
        return result;
    }