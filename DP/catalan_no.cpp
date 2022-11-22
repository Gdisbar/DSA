Total number of possible Binary Search Trees and Binary Trees with n keys
=============================================================================
Total number of possible Binary Search Trees with n different keys 
(countBST(n)) = Catalan number Cn = (2*n)! / ((n + 1)! * n!)

For n = 0, 1, 2, 3, … values of Catalan numbers 
are 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, ….So are numbers of 
Binary Search Trees.

Total number of possible Binary Trees with n different keys 
(countBT(n)) = countBST(n) * n! 

catalan(n) = C(2*n,n)/(n+1) = binomialCoeff(2*n, n) / (n+1)
// TC : k
// Since C(n, k) = C(n, n-k) , we change k = n - k for all k > n - k
// Calculate value of [n*(n-1)*---*(n-(k-1))] / [k*(k-1)*---*1]