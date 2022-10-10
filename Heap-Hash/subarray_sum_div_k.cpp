974. Subarray Sums Divisible by K
===================================
Given an integer array nums and an integer k, return the number of non-empty 
subarrays that have a sum divisible by k.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [4,5,0,-2,-3,1], k = 5
Output: 7
Explanation: There are 7 subarrays with a sum divisible by k = 5:
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]

Example 2:

Input: nums = [5], k = 9
Output: 0


i=      0 1 2  3  4 5
nums =  4 5 0 -2 -3 1 , k=5
sum=    4 4 4  2  4 0
mp	  1 1 2 3  1  4 2  ---> mp[0]=2,mp[2]=1,mp[4]=4 

//TC : n+k

int subarraysDivByK(vector<int>& nums, int k) {
        int n=nums.size();
        int cnt=0,sum=0;
        vector<int> mp(k,0);
        mp[sum]++;
        for(int i=0;i<n;++i){
           sum+=nums[i];
           sum=(sum%k+k)%k;
           cnt+=mp[sum]++;
        }
        return cnt;
    }

// rather than counting at the same time we can do the following

 // for (int i = 0; i < k; i++)
 
 //   // If there are more than one prefix subarrays with a particular mod value.
 //        if (mod[i] > 1)
 //            result += (mod[i] * (mod[i] - 1)) / 2;
 
 // // add the elements which are divisible by k itself
 //    // i.e., the elements whose sum = 0
 //    result += mod[0];
