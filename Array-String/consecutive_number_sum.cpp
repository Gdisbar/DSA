829. Consecutive Numbers Sum
===============================
// Given an integer n, return the number of ways you can write n as the sum of 
// consecutive positive integers.

 

// Example 1:

// Input: n = 5
// Output: 2
// Explanation: 5 = 2 + 3

// Example 2:

// Input: n = 9
// Output: 3
// Explanation: 9 = 4 + 5 = 2 + 3 + 4

// Example 3:

// Input: n = 15
// Output: 4
// Explanation: 15 = 8 + 7 = 4 + 5 + 6 = 1 + 2 + 3 + 4 + 5


// The thought process goes like this- Given a number N, we can possibly write it as 
// a sum of 2 numbers, 3 numbers, 4 numbers and so on. Let''s assume the fist 
// number in this series be x. Hence, we should have

// x + (x+1) + (x+2)+...+ k terms = N
// kx + k*(k-1)/2 = N implies
// kx = N - k*(k-1)/2
// So, we can calculate the RHS for every value of k and if it is a multiple of k then 
// we can construct a sum of N using k terms starting from x.
// Now, the question arises, till what value of k should we loop for? That''s easy. 
// In the worst case, RHS should be greater than 0. That is
// N - k*(k-1)/2 > 0 which implies
// k*(k-1) < 2N which can be approximated to
// k*k < 2N ==> k < sqrt(2N)
// Hence the overall complexity of the algorithm is O(sqrt(N))



class Solution {
public:
    int consecutiveNumbersSum(int N) {
        int count = 1;
        // if we consider N itself a sum of 0+N then k<sqrt(2*N)+1
        for( int k = 2; k < sqrt( 2 * N ); k++ ) {
            if ( ( N - ( k * ( k - 1 )/2) ) % k == 0) count++;
        }
        return count;
    }
};
