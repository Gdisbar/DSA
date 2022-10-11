// https://www.geeksforgeeks.org/tiling-with-dominoes/

Tiling with Dominoes
=======================
// Given a 3 x n board, find the number of ways to fill it with 2 x 1 dominoes.

// Examples : 
 

// Input : 2
// Output : 3

// Input : 8
// Output : 153

// Input : 12
// Output : 2131

// An =  No. of ways to completely fill a 3 x n board. (We need to find this)
// Bn =  No. of ways to fill a 3 x n board with top corner in last column not filled.
// Cn =  No. of ways to fill a 3 x n board with bottom corner in last column not filled.

// Finding Recurrences 
// Note: Even though Bn and Cn are different states, they will be equal for same ‘n’. 
// i.e B[n] = C[n] 
// Hence, we only need to calculate one of them.
// Calculating A[n]: 
// 	A[n] = A[n-2] + B[n-1] + C[n-1] = A[n-2] + 2*B[n-1]
// Calculating B[n]: 
//     B[n] = A[n-1] + B[n-2]

// Base Cases: A[0] = 1,A[1] = 0 ,B[0] = 0, B[1] = 1

int countWays(int n)
{
    int A[n + 1], B[n + 1];
    A[0] = 1, A[1] = 0, B[0] = 0, B[1] = 1;
    for (int i = 2; i <= n; i++) {
        A[i] = A[i - 2] + 2 * B[i - 1];
        B[i] = A[i - 1] + B[i - 2];
    }
 
    return A[n];
}