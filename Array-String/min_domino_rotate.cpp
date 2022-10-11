1007. Minimum Domino Rotations For Equal Row
==============================================
// In a row of dominoes, tops[i] and bottoms[i] represent the top and bottom halves 
// of the ith domino. (A domino is a tile with two numbers from 1 to 6 - one on each 
// half of the tile.)

// We may rotate the ith domino, so that tops[i] and bottoms[i] swap values.

// Return the minimum number of rotations so that all the values in tops are the same, 
// or all the values in bottoms are the same.

// If it cannot be done, return -1.

 

// Example 1:

// Input: tops = [2,1,2,4,2,2], bottoms = [5,2,6,2,3,2]
// Output: 2
// Explanation: 
// The first figure represents the dominoes as given by tops and bottoms: before we 
// do any rotations.
// If we rotate the second and fourth dominoes, we can make every value in the top row 
// equal to 2, as indicated by the second figure.

// Example 2:

// Input: tops = [3,5,1,2,3], bottoms = [3,6,3,3,4]
// Output: -1
// Explanation: 
// In this case, it is not possible to rotate the dominoes to make one row of values 
// equal.

// Finding the maximum occurring element & then changing all other to that will not
// work as there me be k element occur n/k times

Intuition
------------
// One observation is that, if A[0] works, no need to check B[0].
// Because if both A[0] and B[0] exist in all dominoes,
// when you swap A[0] in a whole row,
// you will swap B[0] in a whole at the same time.
// The result of trying A[0] and B[0] will be the same.

Solution 1:
-----------------
// Count the occurrence of all numbers in A and B,
// and also the number of domino with two same numbers.

// Try all possibilities from 1 to 6.
// If we can make number i in a whole row,
// it should satisfy that countA[i] + countB[i] - same[i] = n

// Take example of
// A = [2,1,2,4,2,2]
// B = [5,2,6,2,3,2]

// countA[2] = 4, as A[0] = A[2] = A[4] = A[5] = 2
// countB[2] = 3, as B[1] = B[3] = B[5] = 2
// same[2] = 1, as A[5] = B[5] = 2

// We have countA[2] + countB[2] - same[2] = 6,
// so we can make 2 in a whole row.

// Time O(N), Space O(1)

    def minDominoRotations(self, A, B):
        for x in [A[0],B[0]]:
            if all(x in d for d in zip(A, B)):
                return len(A) - max(A.count(x), B.count(x))
        return -1

def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        for target in [tops[0],bottoms[0]]:
            missingT,missingB=0,0
            for i,pair in enumerate(zip(tops,bottoms)):
                top,bottom=pair
                if not(top==target or bottom==target):
                    break
                if top!=target:missingT+=1
                if bottom!=target:missingB+=1
                if i==len(tops)-1:
                    return min(missingT,missingB)
        return -1
Solution 2
---------------
    // Try make A[0] in a whole row,
    // the condition is that A[i] == A[0] || B[i] == A[0]
    // a and b are the number of swap to make a whole row A[0]

    // Try B[0]
    // the condition is that A[i] == B[0] || B[i] == B[0]
    // a and b are the number of swap to make a whole row B[0]

    // Return -1


int minDominoRotations(vector<int>& A, vector<int>& B) {
        int n = A.size();
        for (int i = 0, a = 0, b = 0; i < n && (A[i] == A[0] || B[i] == A[0]); ++i) {
            if (A[i] != A[0]) a++;
            if (B[i] != A[0]) b++;
            if (i == n - 1) return min(a, b);
        }
        for (int i = 0, a = 0, b = 0; i < n && (A[i] == B[0] || B[i] == B[0]); ++i) {
            if (A[i] != B[0]) a++;
            if (B[i] != B[0]) b++;
            if (i == n - 1) return min(a, b);
        }
        return -1;
    }

Solution 3
---------------
// Find intersection set s of all {A[i], B[i]}
// s.size = 0, no possible result.
// s.size = 1, one and only one result.
// s.size = 2, it means all dominoes are [a,b] or [b,a], try either one.
// s.size > 2, impossible.

    def minDominoRotations(self, A, B):
        s = reduce(set.__and__, (set(d) for d in zip(A, B)))
        if not s: return -1
        x = s.pop()
        return min(len(A) - A.count(x), len(B) - B.count(x))