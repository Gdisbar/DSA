930. Binary Subarrays With Sum
--------------------------------
Given a binary array nums and an integer goal, return the number of non-empty 
subarrays with a sum goal.

A subarray is a contiguous part of the array.

 

Example 1:

Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are bolded and underlined below:
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]

Example 2:

Input: nums = [0,0,0,0,0], goal = 0
Output: 15

Solution 1: HashMap

Count the occurrence of all prefix sum.

I didn''t notice that the array contains only 0 and 1,so this solution also works 
if have negatives.

Space O(N)
Time O(N)


int numSubarraysWithSum(vector<int>& nums, int goal) {
        unordered_map<int,int> mp({{0,1}});
        int psum = 0, ans = 0;
        for(int x : nums){
            psum +=x;
            ans += mp[psum-goal];
            mp[psum]++;
        }
        return ans;
    }



Solution 2: Sliding Window

We have done this hundreds time.
Space O(1)
Time O(N)


    int numSubarraysWithSum(vector<int>& A, int S) {
        return atMost(A, S) - atMost(A, S - 1);
    }

    int atMost(vector<int>& A, int S) {
        if (S < 0) return 0;
        int res = 0, i = 0, n = A.size();
        for (int j = 0; j < n; j++) {
            S -= A[j];
            while (S < 0)
                S += A[i++];
            res += j - i + 1;
        }
        return res;
    }