//for(int i = 0; i < weights.size(); i++){
//             sum += weights[i];
//             if(sum > capacity){
//                 sum = 0;
//                 i--;
//                 res++;
//             }
//         }
while(left < right){
    int mid = left + (right-left) / 2;
    int days = conveyDays(weights, mid); //sum>mid (after sum reset) or days> D 
    
    if(days > D){ // the capacity is too small
        left = mid+1;
    }else{ //the capacity is too large or can meet condition, but we need to try smaller one
        right = mid; 
    }
}
return left;


// exact same as above ,  similar part split into m non-empty continuous subarrays 
// + find minimize/maximize largest/smallest sum among the subarrays , accordingly
// lo & ispossible() will change


1011. Capacity To Ship Packages Within D Days
===================================================
A conveyor belt has packages that must be shipped from one port to another 
within days days.The i-th package on the conveyor belt has a weight of weights[i]. 
Each day, we load the ship with packages on the conveyor belt (in the order given by 
weights). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on 
the conveyor belt being shipped within days days.

 // Translation : split the array into d  non-empty continuous subarrays and 
//  minimize the largest sum(capacity) among these d subarrays(to ship all packages)

Example 1:

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 
days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 
14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) 
is not allowed. -->// basically means continuous subarray

Example 2:

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: A ship capacity of 6 is the minimum to ship all the packages in 3 days 
like this:
1st day: 3, 2
2nd day: 2, 4
3rd day: 1, 4

Example 3:

Input: weights = [1,2,3,1,1], days = 4
Output: 3
Explanation:
1st day: 1
2nd day: 2
3rd day: 3
4th day: 1, 1

// 97% faster , 94% less memory
//TC : n(for finding max & sum) + n*log(n) [sum calculation * inside binary search]

class Solution {
private:
    bool ispossible(vector<int>& weights,int m,int days){
        int d=1,s=0;
        for(int i=0;i<weights.size();++i){
            s+=weights[i];
            if(s>m){ //need to change date
                s=weights[i];
                d++;
                if(s>m) return false; //if at any instance we find that this value is greater than current max sum we are checking for we return false;
            }
            if (d>m) //if we need more than required partitions for the current max ("mid") value we are checking for we return false;
                return false;
        }
        return d<=days; //you can retuen true here
        //return true; // if everything is fine without violating any condition we return true;
    }
public:
    int shipWithinDays(vector<int>& weights, int days) {
        // int l=*max_element(weights.begin(),weights.end());
        // int h=accumulate(weights.begin(),weights.end(),0);
        int l=0,h=0;
        for(int i=0;i<weights.size();++i){
        	if (weights[i]==INT_MAX) return INT_MAX; //Even if you do n partitions you won't be able to reduce this, so simply this will be the max sum.
            l=max(l,weights[i]);
            h+=weights[i];
        }
        if(weights.size()==days){return l;} // need to keep 1 package everyday
        int ans=0;
        while(l<=h){
            int m=l+(h-l)/2;
        //can we shift all packages in  given days by keeping <=m  weight package everyday ?
            if(ispossible(weights,m,days)){ 
                ans=m; // the capacity can meet condition, but we need to try smaller one
                h=m-1; // the capacity is too large,search for lower answer
            }
            else l=m+1; // the capacity is too small,search for higher load
        }
        return ans;
    }
};

//same idea but different way of implementation

// class Solution {
// public:
//     int conveyDays(vector<int>& weights, int capacity){
//         int res = 0;
//         int sum = 0;
//         for(int i = 0; i < weights.size(); i++){
//             sum += weights[i];
//             if(sum > capacity){
//                 sum = 0;
//                 i--;
//                 res++;
//             }
//         }
//         return res+1;
//     }
    
//     int shipWithinDays(vector<int>& weights, int D) {
//         int left = -1, right = 0;
//         for(int i = 0; i < weights.size(); i++){
//             left = max(left, weights[i]);
//             right += weights[i];
//         }
//         while(left < right){
//             int mid = left + (right-left) / 2;
//             int days = conveyDays(weights, mid);
//             // the capacity is too small
//             if(days > D){
//                 left = mid+1;
//             }else{ // // the capacity is too large or can meet condition, but we need to try smaller one
//                 right = mid; 
//             }
//         }
//         return left;
       
//     }
// };


410. Split Array Largest Sum / book allocation problem
================================================================
Given an array nums which consists of non-negative integers and an integer stud, 
you can split the array into m non-empty continuous subarrays.

Write an algorithm to minimize the largest sum among these m subarrays.

 

Example 1:

Input: nums = [7,2,5,10,8], m = 2
Output: 18
Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.

Example 2:

Input: nums = [1,2,3,4,5], m = 2
Output: 9

Example 3:

Input: nums = [1,4,4], m = 3
Output: 4

// Exact same as above 


410. Split Array Largest Sum
==============================
Given an array nums which consists of non-negative integers and an integer m, 
you can split the array into m non-empty continuous subarrays.

Write an algorithm to minimize the largest sum among these m subarrays.

 

Example 1:

Input: nums = [7,2,5,10,8], m = 2
Output: 18
Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.

Example 2:

Input: nums = [1,2,3,4,5], m = 2
Output: 9

Example 3:

Input: nums = [1,4,4], m = 3
Output: 4

