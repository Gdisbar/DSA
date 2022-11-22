Maximum sum of absolute difference of any permutation
==========================================================
Input : { 1, 2, 4, 8 }
Output : 18
Explanation : For the given array there are 
several sequence possible
like : {2, 1, 4, 8}
       {4, 2, 1, 8} and some more.
Now, the absolute difference of an array sequence will be
like for this array sequence {1, 2, 4, 8}, the absolute
difference sum is 
= |1-2| + |2-4| + |4-8| + |8-1|
= 14
For the given array, we get the maximum value for
the sequence {1, 8, 2, 4}
= |1-8| + |8-2| + |2-4| + |4-1|
= 18


sort array 
take max & min together & take their absolute difference i.e a[i]&a[n-1-i]

Minimum sum of absolute difference of pairs of two arrays
==========================================================
sort array 
take consecutive pairs i.e a[i]&a[i+1]

Swap and Maximize
====================
Given circular array find Maximum sum of absolute difference of any permutation
Input:
N = 4
a[] = {4, 2, 1, 8}
Output: 
18
Explanation: Rearrangement done is {1, 8, 
2, 4}. Sum of absolute difference between 
consecutive elements after rearrangement = 
|1 - 8| + |8 - 2| + |2 - 4| + |4 - 1| = 
7 + 6 + 2 + 3 = 18.

sort array 
// Subtracting a1, a2, a3,....., a(n/2)-1, an/2
// twice and adding a(n/2)+1, a(n/2)+2, a(n/2)+3,.
// ...., an - 1, an twice.
for (int i = 0; i < n/2; i++){
        sum -= (2 * arr[i]);
        sum += (2 * arr[n - i - 1]);
}

Smallest subset with sum greater than all other elements
==========================================================
Input : arr[] = {3, 1, 7, 1}
Output : 1
Smallest subset is {7}. Sum of
this subset is greater than all
other elements {3, 1, 1}

Input : arr[] = {2, 1, 2}
Output : 2
In this example one element is not 
enough. We can pick elements with 
values 1, 2 or 2, 2. In any case, 
the minimum count is 2.

calculate sum of the array 
sort in descending order + count number of element upto check curr_sum > sum/2