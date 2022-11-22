29. Divide Two Integers
=========================
// Given two integers dividend and divisor, divide two integers without using 
// multiplication, division, and mod operator.

// The integer division should truncate toward zero, which means losing its 
// fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would 
// be truncated to -2.

// Return the quotient after dividing dividend by divisor.

// Note: Assume we are dealing with an environment that could only store integers 
// within the 32-bit signed integer range: [−231, 231 − 1]. For this problem, 
// if the quotient is strictly greater than 231 - 1, then return 231 - 1, and if 
// the quotient is strictly less than -231, then return -231.

 

// Example 1:

// Input: dividend = 10, divisor = 3
// Output: 3
// Explanation: 10/3 = 3.33333.. which is truncated to 3.

// Example 2:

// Input: dividend = 7, divisor = -3
// Output: -2
// Explanation: 7/-3 = -2.33333.. which is truncated to -2.



// The description note that:
// "Assume we are dealing with an environment,
// which could only store integers within the 32-bit signed 
// integer range: [−2^31, 2^31 − 1]."

// But most of solution use "long" integer.
// So I share my solution here.

// Solution 1

// Only one corner case is -2^31 / 1 and I deal with it at the first line.

// This solution has O(logN^2) time complexity.



int divide(int A, int B) {
    if (A == INT_MIN && B == -1) return INT_MAX;
    int a = abs(A), b = abs(B), res = 0, x = 0;
    while (a - b >= 0) {
        for (x = 0; a - (b << x << 1) >= 0; x++);
        res += 1 << x;
        a -= b << x;
    }
    return (A > 0) == (B > 0) ? res : -res;
}

// O(32)
 int divide(int A, int B) {
        if (A == INT_MIN && B == -1) return INT_MAX;
        int a = abs(A), b = abs(B), res = 0;
        for (int x = 31; x >= 0; x--)
            if ((signed)((unsigned)a >> x) - b >= 0)
                res += 1 << x, a -= b << x;
        return (A > 0) == (B > 0) ? res : -res;
    }