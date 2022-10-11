Find the number of jumps to reach X in the number line from zero
====================================================================
// Given an integer X. The task is to find the number of jumps to reach a point 
// X in the number line starting from zero. 
// Note: The first jump made can be of length one unit and each successive jump 
// will be exactly one unit longer than the previous jump in length. It is allowed 
// to go either left or right in each jump. 

// Examples: 
 

// Input : X = 8
// Output : 4
// Explanation : 
// 0 -> -1 -> 1 -> 4-> 8 are possible stages.

// Input : X = 9
// Output : 5
// Explanation : 
// 0 -> -1 -> -3 -> 0 -> 4-> 9 are possible stages

// Approach: On observing carefully, it can be said easily that: 
 

// If you have always jumped in the right direction then after n jumps you will be 
// at the point p = 1 + 2 + 3 + 4 + … + n.
// In any of these n jumps, if instead of jumping right, you jumped left in the 
// kth jump (k<=n), you would be at point p – 2k.
// Moreover, by carefully choosing which jumps to go left and which to go right, 
// after n jumps, you can be at any point between 
// [ – (n * (n + 1) / 2) , n * (n + 1) / 2] with the same parity as n * (n + 1) / 2.

// Keeping the above points in mind, what you must do is simulate the jumping process, 
// always jumping to the right, and if at some point, you’ve reached a point that has 
// the same parity as X and is at or beyond X, you’ll have your answer.

// TC : n
// Utility function to calculate sum
// of numbers from 1 to x
int getsum(int x)
{
    return (x * (x + 1)) / 2;
}
 
// Function to find the number of jumps
// to reach X in the number line from zero
int countJumps(int n)
{
    // First make number positive
    // Answer will be same either it is
    // Positive or negative
    n = abs(n);
 
    // To store required answer
    int ans = 0;
 
    // Continue till number is lesser or not in same parity
    while (getsum(ans) < n or (getsum(ans) - n) & 1)
        ans++;
 
    // Return the required answer
    return ans;
}