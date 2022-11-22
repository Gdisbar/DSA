39. Combination Sum
=====================
// Given an array of distinct integers candidates and a target integer target, 
// return a list of all unique combinations of candidates where the chosen 
// numbers sum to target. You may return the combinations in any order.

// The same number may be chosen from candidates an unlimited number of times. 
// Two combinations are unique if the frequency of at least one of the chosen 
// numbers is different.

// It is guaranteed that the number of unique combinations that sum up to target
// is less than 150 combinations for the given input.

 

// Example 1:

// Input: candidates = [2,3,6,7], target = 7
// Output: [[2,2,3],[7]]
// Explanation:
// 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple 
// times.
// 7 is a candidate, and 7 = 7.
// These are the only two combinations.

// Example 2:

// Input: candidates = [2,3,5], target = 8
// Output: [[2,2,2,2],[2,3,3],[3,5]]

// Example 3:

// Input: candidates = [2], target = 1
// Output: []

class Solution {
public:
    vector<vector<int> > combinationSum(vector<int> &candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int> > res;
        vector<int> combination;
        helper(candidates, target, res, combination, 0);
        return res;
    }
private:
    void helper(vector<int> &candidates,int target,vector<vector<int> > &res,
                    vector<int> &combination,int begin) {
        if (!target) {
            res.push_back(combination);
            return;
        }
        for (int i = begin; i != candidates.size() && target >= candidates[i]; ++i) {
            combination.push_back(candidates[i]);
            helper(candidates, target - candidates[i], res, combination, i);
            combination.pop_back();
        }
    }
};


40. Combination Sum II
=======================
// Given a collection of candidate numbers (candidates) and a target 
// number (target), find all unique combinations in candidates where the 
// candidate numbers sum to target.

// Each number in candidates may only be used once in the combination.

// Note: The solution set must not contain duplicate combinations.

 

// Example 1:

// Input: candidates = [10,1,2,7,6,1,5], target = 8
// Output: 
// [
// [1,1,6],
// [1,2,5],
// [1,7],
// [2,6]
// ]

// Example 2:

// Input: candidates = [2,5,2,1,2], target = 5
// Output: 
// [
// [1,2,2],
// [5]
// ]

class Solution {
public:
    vector<vector<int> > combinationSum2(vector<int> &candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int> > res;
        vector<int> combination;
        helper(candidates, target, res, combination, 0);
        return res;
    }
private:
    void helper(vector<int> &candidates,int target,vector<vector<int> > &res,
                    vector<int> &combination,int begin) {
        if (!target) {
            res.push_back(combination);
            return;
        }
        for (int i = begin; i != candidates.size() && target >= candidates[i]; ++i) {
            if (i == begin || candidates[i] != candidates[i - 1]) {
                combination.push_back(candidates[i]);
                helper(candidates, target - candidates[i], res, combination, i);
                combination.pop_back();
            }
        }
    }
};



216. Combination Sum III
===========================
// Find all valid combinations of k numbers that sum up to n such that the 
// following conditions are true:

//     Only numbers 1 through 9 are used.
//     Each number is used at most once.

// Return a list of all possible valid combinations. The list must not 
// contain the same combination twice, and the combinations may be returned in 
// any order.

 

// Example 1:

// Input: k = 3, n = 7
// Output: [[1,2,4]]
// Explanation:
// 1 + 2 + 4 = 7
// There are no other valid combinations.

// Example 2:

// Input: k = 3, n = 9
// Output: [[1,2,6],[1,3,5],[2,3,4]]
// Explanation:
// 1 + 2 + 6 = 9
// 1 + 3 + 5 = 9
// 2 + 3 + 4 = 9
// There are no other valid combinations.

// Example 3:

// Input: k = 4, n = 1
// Output: []
// Explanation: There are no valid combinations.
// Using 4 different numbers in the range [1,9], the smallest sum we 
// can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.


class Solution {
public:
    vector<vector<int> > combinationSum3(int k, int n) {
        vector<vector<int> > res;
        vector<int> combination;
        combinationSum3(n, res, combination, 1, k);
        return res;
    }
private:
    void helper(int target,vector<vector<int> > &res,vector<int> &combination, 
                    int begin, int need) {
        if (!target) {
            res.push_back(combination);
            return;
        }
        else if (!need)
            return;
        for (int i=begin; i!=10 && target>=i*need+need*(need-1)/2; ++i) {
            combination.push_back(i);
            helper(target - i, res, combination, i + 1, need - 1);
            combination.pop_back();
        }
    }
};


377. Combination Sum IV
===========================
// Given an array of distinct integers nums and a target integer target, 
// return the number of possible combinations that add up to target.

// The test cases are generated so that the answer can fit in a 32-bit integer.

 

// Example 1:

// Input: nums = [1,2,3], target = 4
// Output: 7
// Explanation:
// The possible combination ways are:
// (1, 1, 1, 1)
// (1, 1, 2)
// (1, 2, 1)
// (1, 3)
// (2, 1, 1)
// (2, 2)
// (3, 1)
// Note that different sequences are counted as different combinations.

// Example 2:

// Input: nums = [9], target = 3
// Output: 0


// So we know that target is the sum of numbers in the array. 
// Imagine we only need one more number to reach target, this number can be any 
// one in the array, right? So the # of combinations of target, 
// comb[target] = sum(comb[target - nums[i]]), where 0 <= i < nums.length, 
// and target >= nums[i].

// In the example given, we can actually find the # of combinations of 4 
// with the # of combinations of 3(4 - 1), 2(4- 2) and 1(4 - 3). As a result, 
// comb[4] = comb[4-1] + comb[4-2] + comb[4-3] = comb[3] + comb[2] + comb[1].

// Then think about the base case. Since if the target is 0, there is 
// only one way to get zero, which is using 0, we can set comb[0] = 1.

// EDIT: The problem says that target is a positive integer that makes 
// me feel it's unclear to put it in the above way. Since target == 0 only 
// happens when in the previous call, target = nums[i], we know that this is 
// the only combination in this case, so we return 1.

// Now we can come up with at least a recursive solution.

public int combinationSum4(int[] nums, int target) {
    if (target == 0) {
        return 1;
    }
    int res = 0;
    for (int i = 0; i < nums.length; i++) {
        if (target >= nums[i]) {
            res += combinationSum4(nums, target - nums[i]);
        }
    }
    return res;
}

// top-down dp

int helper(vector<int>&nums, int target) {
    if(target==0) return 1; // dp[0]=1 , dp[target+1]
    // 0 means no combination sum for target, so -1
    if (dp[target] != -1) {
        return dp[target];
    }
    int res = 0;
    for (int i = 0; i < nums.length; i++) {
        if (target >= nums[i]) {
            res += helper(nums, target - nums[i]);
        }
    }
    dp[target] = res;
    return res;
}

// bottom-up dp

public int combinationSum4(int[] nums, int target) {
    int[] comb = new int[target + 1];
    comb[0] = 1;
    for (int i = 1; i < comb.length; i++) {
        for (int j = 0; j < nums.length; j++) {
            if (i - nums[j] >= 0) {
                comb[i] += comb[i - nums[j]];
            }
        }
    }
    return comb[target];
}