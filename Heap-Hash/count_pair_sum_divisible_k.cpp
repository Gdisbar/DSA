Count pairs in array whose sum is divisible by K
=====================================================
Given an array A[] and positive integer K, the task is to count the 
total number of pairs in the array whose sum is divisible by K. 


Examples: 

Input : A[] = {2, 2, 1, 7, 5, 3}, K = 4
Output : 5
Explanation : 
There are five pairs possible whose sum
is divisible by '4' i.e., (2, 2), 
(1, 7), (7, 5), (1, 3) and (5, 3)

Input : A[] = {5, 9, 36, 74, 52, 31, 42}, K = 3
Output : 7

// Program to count pairs whose sum divisible
// by 'K'
int countKdivPairs(int A[], int n, int K)
{
    // Create a frequency array to count
    // occurrences of all remainders when
    // divided by K
    int freq[K] = { 0 };
 
    // Count occurrences of all remainders
    for (int i = 0; i < n; i++)
        ++freq[A[i] % K];
 
    // If both pairs are divisible by 'K'
    int sum = freq[0] * (freq[0] - 1) / 2;
 
    // count for all i and (k-i)
    // freq pairs
    for (int i = 1; i <= K / 2 && i != (K - i); i++)
        sum += freq[i] * freq[K - i];
    // If K is even
    if (K % 2 == 0)
        sum += (freq[K / 2] * (freq[K / 2] - 1) / 2);
    return sum;
}
 