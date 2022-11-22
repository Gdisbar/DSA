Chocolate Distribution Problem
===============================
// Given an array A[ ] of positive integers of size N, where each value represents the
// number of chocolates in a packet. Each packet can have a variable number of 
// chocolates. There are M students, the task is to distribute chocolate packets among 
// M students such that :
// 1. Each student gets exactly one packet.
// 2. The difference between maximum number of chocolates given to a student and minimum 
//    number of chocolates given to a student is minimum.

// Example 1:

// Input:
// N = 8, M = 5
// A = {3, 4, 1, 9, 56, 7, 9, 12}
// Output: 6
// Explanation: The minimum difference between 
// maximum chocolates and minimum chocolates 
// is 9 - 3 = 6 by choosing following M packets :
// {3, 4, 9, 7, 9}.

// Example 2:

// Input:
// N = 7, M = 3
// A = {7, 3, 2, 4, 9, 12, 56}
// Output: 2
// Explanation: The minimum difference between
// maximum chocolates and minimum chocolates
// is 4 - 2 = 2 by choosing following M packets :
// {3, 2, 4}.

long long findMinDiff(vector<long long> a, long long n, long long m){
        sort(a.begin(),a.end());
        long long mn = LONG_MAX;
        for(long long i = 0; i <=n-m ;++i){
            mn=min(mn,a[i+m-1]-a[i]);
        }
        return mn;
    } 



000000000
============
// There are n children standing in a line. Each child is assigned a rating value 
// given in the integer array ratings.

// You are giving candies to these children subjected to the following requirements:

//     Each child must have at least one candy.
//     Children with a higher rating get more candies than their neighbors.

// Return the minimum number of candies you need to have to distribute the candies 
// to the children.

 

// Example 1:

// Input: ratings = [1,0,2]
// Output: 5
// Explanation: You can allocate to the first, second and third child with 2, 1, 2 
// candies respectively.

// Example 2:

// Input: ratings = [1,2,2]
// Output: 4
// Explanation: You can allocate to the first, second and third child with 1, 2, 1 
// candies respectively.
// The third child gets 1 candy because it satisfies the above two conditions.

// TC : n , SC : n

vector<int> left(n,1),right(n,1);
for(int i=1;i<n;++i){
	if(ratings[i]>ratings[i-1])
        left[i]=left[i-1]+1;
}
for(int i=n-2;i>=0;--i){
    if(ratings[i]>ratings[i+1])
        right[i]=right[i+1]+1;
}
for(int i=0;i<n;++i){
    res+=max(left[i],right[i]);
}

// fastest but not memory optimized

int candy(vector<int>& r) {
        int n = r.size();
        vector<int> c(n, 1);
        for (int i = 1; i < n; i++)
            if (r[i] > r[i - 1]) c[i] = c[i - 1] + 1;
        for (int i = n - 2; ~i; i--)
            if (r[i] > r[i + 1]) c[i] = max(c[i], c[i + 1] + 1);
        int res = 0;
        for (auto t: c) res += t;
        return res;
    }

// O(1) Space Approach :

// We can consider this problem like valley and peak problem. In each valley 
// there should be 1 candy and for each increasing solpe in either side we need to 
// increse candy by 1. Peaks have highest candy. If any equal rating is found then 
// candy resets to 1 as two equal neighbours may have any number of candies. The peak 
// should contain the higher number of candy between which is calculated from the 
// incresing slope and which is calculated from decreasing slope. Because this 
// will satisfy the condition that peak element is having more candies than its 
// neighbours.

// Example :

// Let take the Rating as : [1,3,6,8,9,5,3,6,8,5,4,2,2,3,7,7,9,8,6,6,6,4,2]

// Each child represented as rating(candy he is given)
// Peak = max(peak, valley)

// See when peak is encountered we take max of the peak calculated from left and 
// valley calculated from right.
// When we get any equal element it gets reset to 1 candy or if it is peak we 
// take max(0, right valley)

//            (5)         (4)                         (3)
//             9           8                           9
//            /|\         /|\                         /|\
//           / | \       / | \                (3)    / | \
//       (4)8  |  5(2)  6  |  5(3)             7 __ 7  |  8(2)
//         /   |   \   (2) |   \              /|   (1) |   \
//        /    |    \ /    |    \            / |    |  |    \         (3)
//    (3)6     |     3     |     4(2)       3(2)    |  |     6 __ 6 __ 6    -> Total candy = 50
//      /      |    (1)    |      \        /   | Reset |    (1)  (1)   |\
//     /       |           |       \      /    |  to 1 |          |    | \
// (2)3        |           |        2 __ 2     |       |          |    |  4(2)
//   /         |           |       (1)  (1)    |       |        Reset  |   \
//  /          |           |                   |       |         to 1  |    \
// 1(1)        |           |                   |       |               |     2(1)
//    Peak= max(5,3)  Peak= max(3,4)    Peak= max(3,0) |         Peak= max(0,3)
//                                                Peak= max(2,3)   



// See the example for better understanding :

// In our code we increase the peak and add peak value until we get the minimum 
// actual peak.
// In case of the decreasing part take this example.

// [7,5,3,2], initial candy = 4, In each iteration valley++ and candy += valley

// 7 starting (valley = 0, candy = 4, candy configuration  = [1,1,1,1])
//  \
//   5 (valley = 1, candy = 4+1 = 5, candy configuration  = [2,1,1,1])
//    \
//     3 (valley = 2, candy = 5+2 = 7, candy configuration = [3,2,1,1])
//      \
//       2 [valley = 3, candy = 7+3 = 10, candy configuration = [4,3,2,1])
	  
// Here see the valley is at last equal to the minimum previous peak value.

// As we have given 1 candy to all before so the peak and valley values are 
// actually 1 less than the actual candy they should get.


int candy(vector<int>& ratings) {
        int n = ratings.size();
        int candy = n, i=1;
        while(i<n){
            if(ratings[i] == ratings[i-1]){ //plateau, reset
                i++; 
                continue;
            }
            
            //For increasing slope , height at left side
            int peak = 0;
            while(ratings[i] > ratings [i-1]){ 
                // finding new peak
                peak++;
                candy += peak;
                i++;
                if(i == n) return candy;
            }
            
            //For decreasing slope ,height at right side
            int valley = 0;
            while(i<n && ratings[i] < ratings[i-1]){
                //depth of valley, minimum height of the previous peak
                valley++;
                candy += valley;
                i++;
            }
            candy -= min(peak, valley); //Keep only the higher peak
        }
        return candy;
    }