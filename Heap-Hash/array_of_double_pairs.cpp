954. Array of Doubled Pairs
============================
// Given an integer array of even length arr, return true if it is possible 
// to reorder arr such that arr[2 * i + 1] = 2 * arr[2 * i] for every 
// 0 <= i < len(arr) / 2, or false otherwise.

 

// Example 1:

// Input: arr = [3,1,3,6]
// Output: false

// Example 2:

// Input: arr = [2,1,2,6]
// Output: false

// Example 3:

// Input: arr = [4,-2,2,-4]
// Output: true
// Explanation: We can take two groups, [-2,-4] and [2,4] to form [-2,-4,2,4] 
// or [2,4,-2,-4].


// why sort ? x is smallest

// If x is a positive number, then it pairs with y = x*2, 
// for example: x = 4 pair with y = 8.
// If x is a non-positive number, then it pairs with y = x/2, 
// for example: x = -8 pair with y = -4.
// If there is no corresponding y then it''s IMPOSSIBLE, return FALSE.

// For example: arr = [2, 4, 1, 8]

// If we process x = 2 first, then there are 2 choices, either 4 or 1 can 
// be paired with 2, if we choose 4 -> we got WRONG ANSWER.
// Because 8 needs 4, so 2 should be paired with 1.

// When a pair of (x and y) match, we need to decrease their count. 
// So we need to use a HashTable data structure to count the frequency of 
// elements in the arr array.

// Here a[2*i+1],a[2*i] --> a[x+1],a[x]

//slower
    // Time: O(NlogN), where N <= 3 * 10^4 is number of elements in arr array.
    // Space: O(N)


class Solution {
public:
    bool canReorderDoubled(vector<int>& arr) {
        sort(arr.begin(), arr.end());
        unordered_map<int, int> cnt;
        for (int x : arr) cnt[x]++;
        
        for (int x : arr) {
            if (cnt[x] == 0) continue;
            if (x < 0 && x % 2 != 0) return false; // For example: arr=[-5, -2, 1, 2], x = -5, there is no x/2 pair to match
            int y = x > 0 ? x*2 : x/2;
            if (cnt[y] == 0) return false; // Don't have the corresponding `y` to match with `x` -> Return IMPOSSIBLE!
            cnt[x]--;
            cnt[y]--;
        }
        return true;
    }
};
//faster
//Python, O(N + KlogK), 100~200ms , 
//for c++ use separate vector to store keys then operate on those values

    def canReorderDoubled(self, A):
        c = collections.Counter(A)
        for x in sorted(c, key=abs):
            if c[x] > c[2 * x]:
                return False
            c[2 * x] -= c[x]
        return True
