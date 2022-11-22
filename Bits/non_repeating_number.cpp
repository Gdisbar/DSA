136. Single Number
====================
// Given a non-empty array of integers nums, every element appears twice 
// except for one. Find that single one.

// You must implement a solution with a linear runtime complexity and use 
// only constant extra space.

 

// Example 1:

// Input: nums = [2,2,1]
// Output: 1

// Example 2:

// Input: nums = [4,1,2,1,2]
// Output: 4

// Example 3:

// Input: nums = [1]
// Output: 1


int singleNumber(vector<int>& nums) { 
       int ans=0;
       //the elements with frequency=2 will result in 0. And then the only 
       //element with frequency=1 will generate the answer.
	   for(auto x:nums)
	   ans^=x;
	   return ans;
    }


137. Single Number II
=========================
// Given an integer array nums where every element appears three times except for one, 
// which appears exactly once. Find the single element and return it.

// You must implement a solution with a linear runtime complexity and use only 
// constant extra space.

 

// Example 1:

// Input: nums = [2,2,3,2]
// Output: 3

// Example 2:

// Input: nums = [0,1,0,1,0,1,99]
// Output: 99

https://leetcode.com/problems/single-number-ii/discuss/43295/Detailed-explanation-and-generalization-of-the-bitwise-operation-method-for-single-numbers

// for (int i : nums) {
//     xm ^= (xm-1 & ... & x1 & i);
//     xm-1 ^= (xm-2 & ... & x1 & i);
//     .....
//     x1 ^= i;
    
//     mask = ~(y1 & y2 & ... & ym) 
//     where yj = xj if kj = 1, and yj = ~xj if kj = 0 (j = 1 to m).

//     xm &= mask;
//     ......
//     x1 &= mask;
// }

// Here is a list of few quick examples to show how the algorithm works:

//     k = 2, p = 1
//     k is 2, then m = 1, we need only one 32-bit integer(x1) as the counter. 
//     And 2^m = k so we do not even need a mask! A complete java program will 
//     look like:

    public int singleNumber(int[] A) {
         int x1 = 0;
         
         for (int i : A) {
             x1 ^= i;
         }
         
         return x1;
    }

    // k = 3, p = 1
    // k is 3, then m = 2, we need two 32-bit integers(x2, x1) as the counter. 
    // And 2^m > k so we do need a mask. Write k in its binary form: k = '11', 
    // then k1 = 1, k2 = 1, so we have mask = ~(x1 & x2). 
    // A complete java program will look like:

    public int singleNumber(int[] A) {
         int x1 = 0, x2 = 0, mask = 0;
   
         for (int i : A) {
             x2 ^= x1 & i;
             x1 ^= i;
             mask = ~(x1 & x2);
             x2 &= mask;
             x1 &= mask;
         }

         return x1;  // p = 1, in binary form p = '01', then p1 = 1, so we should return x1; 
                     // if p = 2, in binary form p = '10', then p2 = 1, so we should return x2.
    }

    // k = 5, p = 3
    // k is 5, then m = 3, we need three 32-bit integers(x3, x2, x1) as the counter. 
    // And 2^m > k so we need a mask. Write k in its binary form: k = '101', 
    // then k1 = 1, k2 = 0, k3 = 1, so we have mask = ~(x1 & ~x2 & x3). 
    // A complete java program will look like:

    public int singleNumber(int[] A) {
         int x1 = 0, x2 = 0, x3  = 0, mask = 0;
   
         for (int i : A) {
             x3 ^= x2 & x1 & i;
             x2 ^= x1 & i;
             x1 ^= i;
             mask = ~(x1 & ~x2 & x3);
             x3 &= mask;
             x2 &= mask;
             x1 &= mask;
         }
       
         return x1;  // p = 3, in binary form p = '011', then p1 = p2 = 1, so we can
                     // return either x1 or x2. But if p = 4, in binary form p = '100', 
                     // only p3 = 1, which implies we can only return x3.
    }



int singleNumber(int* nums, int numsSize, int k) //k>=2
{
    int counter[32];
    int i, j;
    int res = 0;
    
    for(i=0; i<32; i++)
        counter[i] = 0;
        
    for(i=0; i<numsSize; i++)
    {
        for(j=0; j<32; j++)
        {
            if(nums[i] & (1<<j))
                counter[j]++;
        }
    }
    
    for(i=0; i<32; i++)
    {
        if(counter[i]%k)
            res |= 1<<i;
    }
    
    return res;
}


260. Single Number III
==========================
// Given an integer array nums, in which exactly two elements appear only 
// once and all the other elements appear exactly twice. Find the two elements 
// that appear only once. You can return the answer in any order.

// You must write an algorithm that runs in linear runtime complexity and uses 
// only constant extra space.

 

// Example 1:

// Input: nums = [1,2,1,3,2,5]
// Output: [3,5]
// Explanation:  [5, 3] is also a valid answer.

// Example 2:

// Input: nums = [-1,0]
// Output: [-1,0]

// Example 3:

// Input: nums = [0,1]
// Output: [1,0]


//     In the first pass, we XOR all elements in the array, and get the XOR of the 
//     two numbers we need to find. Note that since the two numbers are distinct, 
//     so there must be a set bit (that is, the bit with value '1') in the XOR 
//     result. Find
//     out an arbitrary set bit (for example, the rightmost set bit).

//     In the second pass, we divide all numbers into two groups, one with the 
//     aforementioned bit set, another with the aforementinoed bit unset. Two 
//     different numbers we need to find must fall into thte two distrinct groups. 
//     XOR numbers in each group, we can find a number in either group.

// Complexity:

//     Time: O (n)

//     Space: O (1)

// A Corner Case:

//     When diff == numeric_limits<int>::min(), -diff is also 
//     numeric_limits<int>::min(). Therefore, the value of diff after 
//     executing diff &= -diff is still numeric_limits<int>::min(). The answer 
//     is still correct.




    vector<int> singleNumber(vector<int>& nums) 
    {
        // Pass 1 : 
        // Get the XOR of the two numbers we need to find
        int diff = accumulate(nums.begin(), nums.end(), 0, bit_xor<int>());
        // Get its last set bit
        diff &= -diff;

        // Pass 2 :
        vector<int> rets = {0, 0}; // this vector stores the two numbers we will return
        for (int num : nums)
        {
            if ((num & diff) == 0) // the bit is not set
            {
                rets[0] ^= num;
            }
            else // the bit is set
            {
                rets[1] ^= num;
            }
        }
        return rets;
    }

