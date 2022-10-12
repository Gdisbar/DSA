1248. Count Number of Nice Subarrays
---------------------------------------

Given an array of integers nums and an integer k. A continuous subarray is called nice if 
there are k odd numbers on it.

Return the number of nice sub-arrays.

 

Example 1:

Input: nums = [1,1,2,1,1], k = 3
Output: 2
Explanation: The only sub-arrays with 3 odd numbers are [1,1,2,1] and [1,2,1,1].

Example 2:

Input: nums = [2,4,6], k = 1
Output: 0
Explanation: There is no odd numbers in the array.

//count # of subarrays with k odd numbers ---> after replacing all odd with 1 & all even with 0
//we convert the problem --> count # of subarray with sum k

    int numberOfSubarrays(vector<int>& A, int k) {
        return atMost(A, k) - atMost(A, k - 1);
    }

    int atMost(vector<int>& A, int k) {
        int res = 0, i = 0, n = A.size();
        for (int j = 0; j < n; j++) {
            k -= A[j] % 2;
            while (k < 0)
                k += A[i++] % 2;
            res += j - i + 1;
        }
        return res;
    }


Solution II: One pass

Actually it''s same as three pointers.
Though we use count to count the number of even numebers.
Insprired by @yannlecun.

Time O(N) for one pass
Space O(1)

        int numberOfSubarrays(vector<int>& A, int k) {
        int res = 0, i = 0, count = 0, n = A.size();
        for (int j = 0; j < n; j++) {
            if (A[j] & 1)
                --k, count = 0;
            while (k == 0)
                k += A[i++] & 1, ++count;
            res += count;
        }
        return res;
    }


992. Subarrays with K Different Integers
-----------------------------------------
Given an integer array nums and an integer k, return the number of good subarrays of nums.

A good array is an array where the number of different integers in that array is exactly k.

    For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [1,2,1,2,3], k = 2
Output: 7
Explanation: Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], 
[2,3], [1,2,1], [2,1,2], [1,2,1,2]

suppose initial window [a] then subarrays that ends with this element are [a]--> 1
now we expand our window [a,b] then subarrays that ends with this new element 
are [b], [a,b] -->2
now we expand our window [a,b,c] then subarrays that ends with this new element 
are [c], [b, c], [a,b,c] -->3
now we expand our window [a,b,c,d] and let suppose this is not valid window so we 
compress window from left side to make it valid window
[b,c,d] then subarrays that ends with this new element are [d], [c,d], [b,c,d] -->3

You can observe that we are only considering subarrays with new element in it which auto. 
eliminate the counting of duplicate subarrays that we already considered previously.
And surprisingly the number of sub-arrays with this new element in it is equal to the 
length of current window.


int atMostK(vector<int>& A, int K) {
        int i = 0, res = 0;
        unordered_map<int, int> count;
        for (int j = 0; j < A.size(); ++j) {
            //we have encountered A[j] for 1st time then k--, increase count of A[j]
            if (!count[A[j]]++) K--;
            //we have ran out of window, we increase i to find the last occurrence of A[i] 
            //here we are at window [2,3] so we have moved [1,2] -> [2,1] -> [1,3] = total 3 step i=3
            while (K < 0) {
                if (!--count[A[i]]) K++;
                i++;
            }
            res += j - i + 1;
        }
        return res;
    }
    
int subarraysWithKDistinct(vector<int>& nums, int k) {
    return atMostK(nums,k)-atMostK(nums,k-1);
}