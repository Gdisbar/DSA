Matrix Chain Multiplication
=================================
// Given the dimension of a sequence of matrices in an array arr[], 
// where the dimension of the ith matrix is (arr[i-1] * arr[i]), the task is to 
// find the most efficient way to multiply these matrices together such that the 
// total number of element multiplications is minimum.

// Examples:

// Input: arr[] = {40, 20, 30, 10, 30}
// Output: 26000
// Explanation:There are 4 matrices of dimensions 40×20, 20×30, 30×10, 10×30.
// Let the input 4 matrices be A, B, C and D.
// The minimum number of  multiplications are obtained by 
// putting parenthesis in following way (A(BC))D.
// The minimum is 20*30*10 + 40*20*10 + 40*10*30

// Input: arr[] = {1, 2, 3, 4, 3}
// Output: 30
// Explanation: There are 4 matrices of dimensions 1×2, 2×3, 3×4, 4×3. 
// Let the input 4 matrices be A, B, C and D.  
// The minimum number of multiplications are obtained by 
// putting parenthesis in following way ((AB)C)D.
// The minimum number is 1*2*3 + 1*3*4 + 1*4*3 = 30

// Input: arr[] = {10, 20, 30}
// Output: 6000  
// Explanation: There are only two matrices of dimensions 10×20 and 20×30. 
// So there  is only one way to multiply the matrices, cost of which is 10*20*30

//recursive

// Matrix Ai has dimension p[i-1] x p[i]
// for i = 1 . . . n
int MatrixChainOrder(int p[], int i, int j)
{
    if (i == j)
        return 0;
    int k;
    int mini = INT_MAX;
    int count;
 
    // Place parenthesis at different places
    // between first and last matrix,
    // recursively calculate count of multiplications
    // for each parenthesis placement
    // and return the minimum count
    for (k = i; k < j; k++)
    {
        count = MatrixChainOrder(p, i, k)+ MatrixChainOrder(p, k + 1, j)
                + p[i - 1] * p[k] * p[j];
 
        mini = min(count, mini);
    }
 
    // Return minimum count
    return mini;
}

// TC : exponential
// SC : 1

// Matrix Ai has dimension p[i-1] x p[i]
// for i = 1..n
int MatrixChainOrder(int p[], int n)
{
 
    /* For simplicity of the program, one
    extra row and one extra column are
    allocated in m[][]. 0th row and 0th
    column of m[][] are not used */
    int m[n][n];
 
    int i, j, k, L, count;
 
    /* m[i, j] = Minimum number of scalar
    multiplications needed to compute the
    matrix A[i]A[i+1]...A[j] = A[i..j] where
    dimension of A[i] is p[i-1] x p[i] */
 
    // cost is zero when multiplying
    // one matrix.
    for (i = 1; i < n; i++)
        m[i][i] = 0;
 
    // L is chain length.
    for (L = 2; L < n; L++)
    {
        for (i = 1; i < n - L + 1; i++) //begin of chain
        {
            j = i + L - 1; //end of chain
            m[i][j] = INT_MAX;
            for (k = i; k < j ; k++)
            {
                // q = cost/scalar multiplications
                count = m[i][k] + m[k + 1][j]+ p[i - 1] * p[k] * p[j];
                m[i][j] = min(count,m[i][j]);
            }
        }
    }
 
    return m[1][n - 1];
}

// TC : n^3
// SC : n^2

// Assume there are following available method 
// minCost(M1, M2) -> returns min cost of multiplying matrices M1 and M2
// Then, for any chained product of matrices like, 
// M1.M2.M3.M4…Mn 
// min cost of chain = min(minCost(M1, M2.M3…Mn), minCost(M1.M2...Mn-1, Mn)) 
// Now we have two subchains (sub problems) : 
// M2.M3…Mn 
// M1.M2...Mn-1

int MatrixChainOrder(int p[], int n)
{
 
    /* For simplicity of the program, one extra row and one extra column are allocated in
    dp[][]. 0th row and 0th column of dp[][] are not used */
    int dp[n][n];
 
    /* dp[i, j] = Minimum number of scalar multiplications needed to compute the matrix M[i]M[i+1]...M[j]
                = M[i..j] where dimension of M[i] is p[i-1] x p[i] */
                 
    // cost is zero when multiplying one matrix.
    for (int i=1; i<n; i++)
        dp[i][i] = 0;
 
    // Simply following above recursive formula.
    for (int L=1; L<n-1; L++)
    for (int i=1; i<n-L; i++)    
        dp[i][i+L] = min(dp[i+1][i+L] + p[i-1]*p[i]*p[i+L],
                    dp[i][i+L-1] + p[i-1]*p[i+L-1]*p[i+L]);    
     
    return dp[1][n-1];
}

// TC : n^2
// SC : n^2


Printing Matrix Chain Multiplication 
=============================================

// Function for printing the optimal
// parenthesization of a matrix chain product
void printParenthesis(int i, int j, int n,
                      int *bracket, char &name)
{
    // If only one matrix left in current segment
    if (i == j)
    {
        cout << name++;
        return;
    }
 
    cout << "(";
 
    // Recursively put brackets around subexpression
    // from i to bracket[j][i].
    // Note that "*((bracket+j*n)+i)" is similar to
    // bracket[j][i]
    printParenthesis(i, *((bracket+j*n)+i), n,
                     bracket, name);
 
    // Recursively put brackets around subexpression
    // from bracket[j][i] + 1 to i.
    printParenthesis(*((bracket+j*n)+i) + 1, j,
                     n, bracket, name);
    cout << ")";
}
 
// Matrix Ai has dimension p[i-1] x p[i] for i = 1..n
// Please refer below article for details of this
// function
// https://goo.gl/k6EYKj
void matrixChainOrder(int p[], int n)
{
    /* For simplicity of the program, one extra
       row and one extra column are allocated in
        m[][]. 0th row and 0th column of m[][]
        are not used */
    int m[n][n];
 
    /* m[i,j] = Minimum number of scalar multiplications
    needed to compute the matrix A[i]A[i+1]...A[j] =
    A[i..j] where dimension of A[i] is p[i-1] x p[i] */
 
    // cost is zero when multiplying one matrix.
    for (int i=1; i<n; i++)
        m[i][i] = 0;
 
    // L is chain length.
    for (int L=2; L<n; L++)
    {
        for (int i=1; i<n-L+1; i++)
        {
            int j = i+L-1;
            m[i][j] = INT_MAX;
            for (int k=i; k<=j-1; k++)
            {
                // q = cost/scalar multiplications
                int q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j];
                if (q < m[i][j])
                {
                    m[i][j] = q;
 
                    // Each entry m[j,ji=k shows
                    // where to split the product arr
                    // i,i+1....j for the minimum cost.
                    m[j][i] = k;
                }
            }
        }
    }
 
    // The first matrix is printed as 'A', next as 'B',
    // and so on
    char name = 'A';
 
    cout << "Optimal Parenthesization is: ";
    printParenthesis(1, n-1, n, (int *)m, name);
    cout << "\nOptimal Cost is : " << m[1][n-1];
}