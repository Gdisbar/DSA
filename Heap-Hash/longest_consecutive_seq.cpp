128. Longest Consecutive Sequence
===================================
// Given an unsorted array of integers nums, return the length of the 
// longest consecutive elements sequence.

// You must write an algorithm that runs in O(n) time.

// Example 1:

// Input: nums = [100,4,200,1,3,2]
// Output: 4
// Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. 
// Therefore its length is 4.

// Example 2:

// Input: nums = [0,3,7,2,5,8,4,6,0,1]
// Output: 9

// Then go through the numbers. If the number x is the start of a 
// streak (i.e., x-1 is not in the set), then test y = x+1, x+2, x+3, ... and 
// stop at the first number y not in the set. The length of the streak is 
// then simply y-x and we update our global best with that. Since we check 
// each streak only once

// TC : n = insert in set + n = iteration + n = count greater element 
// SC : n = set size

int longestConsecutive(vector<int>& nums) {
    unordered_set<int> s;
    for(int x : nums) s.insert(x);
    int longest=0;
    for(int x : nums){
        if(!s.count(x-1)){
            int xx=x+1;
            while(s.count(xx)){
                xx++;
            }
            longest=max(longest,xx-x);
        }
        
    }
    return longest;
}

 def longestConsecutive(self, nums):
        res, left = 0, set(nums)
        while left:
            l = r = left.pop()
            while l - 1 in left: left.remove(l - 1); l -= 1;
            while r + 1 in left: left.remove(r + 1); r += 1;
            res = max(res, r - l + 1)
        return res