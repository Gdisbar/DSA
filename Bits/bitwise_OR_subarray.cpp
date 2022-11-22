898. Bitwise ORs of Subarrays
===============================
// We have an array arr of non-negative integers.

// For every (contiguous) subarray sub = [arr[i], arr[i + 1], ..., arr[j]] 
// (with i <= j), we take the bitwise OR of all the elements in sub, obtaining 
// a result arr[i] | arr[i + 1] | ... | arr[j].

// Return the number of possible results. Results that occur more than once are 
// only counted once in the final answer

// Example 1:

// Input: arr = [0]
// Output: 1
// Explanation: There is only one possible result: 0.

// Example 2:

// Input: arr = [1,1,2]
// Output: 3
// Explanation: The possible subarrays are [1], [1], [2], [1, 1], [1, 2], [1, 1, 2].
// These yield the results 1, 1, 2, 1, 3, 3.
// There are 3 unique values, so the answer is 3.

// Example 3:

// Input: arr = [1,2,4]
// Output: 6
// Explanation: The possible results are 1, 2, 3, 4, 6, and 7.

// Intuition:

// Assume B[i][j] = A[i] | A[i+1] | ... | A[j]
// Hash set cur stores all wise B[0][i], B[1][i], B[2][i], B[i][i].

// When we handle the A[i+1], we want to update cur
// So we need operate bitwise OR on all elements in cur.
// Also we need to add A[i+1] to cur.

// In each turn, we add all elements in cur to res.

// Complexity:

// Time O(30N)

// Normally this part is easy.
// But for this problem, time complexity matters a lot.

// The solution is straight forward,
// while you may worry about the time complexity up to O(N^2)
// However, it's not the fact.
// This solution has only O(30N)

// The reason is that, B[0][i] >= B[1][i] >= ... >= B[i][i].
// B[0][i] covers all bits of B[1][i]
// B[1][i] covers all bits of B[2][i]
// ....

// There are at most 30 bits for a positive number 0 <= A[i] <= 10^9.
// So there are at most 30 different values for B[0][i], B[1][i], B[2][i], ..., B[i][i].
// Finally cur.size() <= 30 and res.size() <= 30 * A.length()

// In a worst case, A = {1,2,4,8,16,..., 2 ^ 29}
// And all B[i][j] are different and res.size() == 30 * A.length()

// Solution 1: Use HashSet


    int subarrayBitwiseORs(vector<int> A) {
        unordered_set<int> res, cur, cur2;
        for (int i: A) {
            cur2 = {i};
            for (int j: cur) cur2.insert(i|j);
            for (int j: cur = cur2) res.insert(j);
        }
        return res.size();
    }


// Solution2: Use Array List

// The elements between res[left] and res[right] are same as the cur in solution 1.


    int subarrayBitwiseORs(vector<int> A) {
        vector<int> res;
        int left = 0, right;
        for (int a: A) {
            right = res.size();
            res.push_back(a);
            for (int i = left; i < right; ++i) {
                if (res.back() != (res[i] | a)) {
                    res.push_back(res[i] | a);
                }
            }
            left = right;
        }
        return unordered_set(res.begin(), res.end()).size();
    }
