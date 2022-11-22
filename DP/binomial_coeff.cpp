Binomial Coefficient
======================
// The following are the common definitions of Binomial Coefficients. 

//     A binomial coefficient C(n, k) can be defined as the coefficient of 
//     x^k in the expansion of (1 + x)^n.

// C(n, k) = C(n-1, k-1) + C(n-1, k)
// C(n, 0) = C(n, n) = 1

int binomialCoeff(int n, int k)
{
    int C[k + 1];
    memset(C, 0, sizeof(C));
 
    C[0] = 1; // nC0 is 1
 
    for (int i = 1; i <= n; i++)
    {
       
        // Compute next row of pascal triangle using
        // the previous row
        for (int j = min(i, k); j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    return C[k];
}

// Time Complexity: O(n*k) 
// Auxiliary Space: O(k)


// C(n, k) 
// = n! / (n-k)! * k!
// = [n * (n-1) *....* 1]  / [ ( (n-k) * (n-k-1) * .... * 1) * 
//                             ( k * (k-1) * .... * 1 ) ]
// After simplifying, we get
// C(n, k) 
// = [n * (n-1) * .... * (n-k+1)] / [k * (k-1) * .... * 1]

// Also, C(n, k) = C(n, n-k)  
// // r can be changed to n-r if r > n-r 


// Returns value of Binomial Coefficient C(n, k)
int binomialCoeff(int n, int k)
{
    int res = 1;
 
    // Since C(n, k) = C(n, n-k)
    if (k > n - k)
        k = n - k;
 
    // Calculate value of
    // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
 
    return res;
}
 
// Time Complexity: O(r) A loop has to be run from 0 to r. So, 
// the time complexity is O(r).

// Auxiliary Space: O(1) As no extra space is required.

// 1. The general formula of nCr is ( n*(n-1)*(n-2)* … *(n-r+1) ) / (r!). 
// We can directly use this formula to find nCr. But that will overflow out of 
// bound. We need to find nCr mod m so that it doesn’t overflow. We can easily 
// do it with modular arithmetic formula. 

// for the  n*(n-1)*(n-2)* ... *(n-r+1) part we can use the formula,
// (a*b) mod m = ((a % m) * (b % m)) % m

// 2. and for the 1/r! part, we need to find the modular inverse of every number 
// from 1 to r. Then use the same formula above with a modular inverse of 1 to r. 
// We can find modular inverse in O(r) time using  the formula, 

// inv[1] = 1
// inv[i] = − ⌊m/i⌋ * inv[m mod i] mod m
// To use this formula, m has to be a prime.

// In the practice problem, we need to show the answer with modulo 1000000007 
// which is a prime. 


// Function to find binomial
// coefficient
int binomialCoeff(int n, int r)
{
 
    if (r > n)
        return 0;
    long long int m = 1000000007;
    long long int inv[r + 1] = { 0 };
    inv[0] = 1;
    if(r+1>=2)
    inv[1] = 1;
 
    // Getting the modular inversion
    // for all the numbers
    // from 2 to r with respect to m
    // here m = 1000000007
    for (int i = 2; i <= r; i++) {
        inv[i] = m - (m / i) * inv[m % i] % m;
    }
 
    int ans = 1;
 
    // for 1/(r!) part
    for (int i = 2; i <= r; i++) {
        ans = ((ans % m) * (inv[i] % m)) % m;
    }
 
    // for (n)*(n-1)*(n-2)*...*(n-r+1) part
    for (int i = n; i >= (n - r + 1); i--) {
        ans = ((ans % m) * (i % m)) % m;
    }
    return ans;
}

// Time Complexity: O(n+k)

// Auxiliary Space: O(k) 