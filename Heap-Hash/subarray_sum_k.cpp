560. Subarray Sum Equals K
=============================
Given an array of integers nums and an integer k, return the total number of 
subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,1,1], k = 2
Output: 2

Example 2:

Input: nums = [1,2,3], k = 3
Output: 2

//Brute force : generate all subaary (n*n) * find sum of each subarray(n) = n^3

//TLE
int subarraySum(vector<int>& nums, int k) {
        int n=nums.size();
        vector<int> pref(n,0);
        pref[0]=nums[0];
        for(int i=1;i<n;++i){
            pref[i]=pref[i-1]+nums[i];
        }
        int cnt=0;
        for(int i=0;i<n;++i){
            for(int j=i;j<n;++j){
                if(i==0&&pref[j]==k) cnt++;
                else if(i>0&&pref[j]-pref[i-1]==k) cnt++;
            }
        }
        return cnt;
    }

//space optimized but still TLE

// for(int i=0;i<n;++i){
//             sum=0;
//             for(int j=i;j<n;++j){
//                sum+=nums[j];
//                 if(sum==k)cnt++;
//             }
//         }

//95% faster,85% less memory

int subarraySum(vector<int>& nums, int k) {
        int n=nums.size();
        int cnt=0,sum=0;
        unordered_map<int,int> mp;
        mp[sum]++;  //sum of none
        for(int i=0;i<n;++i){
           sum+=nums[i];
           // if(sum==k){
           //     cnt++;
           // }
           if(mp.find(sum-k)!=mp.end()){
               cnt+=mp[sum-k];
           }
           mp[sum]++;
        }
        return cnt;
    }