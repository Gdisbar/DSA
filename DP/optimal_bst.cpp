Optimal Binary Search Tree
============================
// Given a sorted array key [0.. n-1] of search keys and an array freq[0.. n-1] 
// of frequency counts, where freq[i] is the number of searches for keys[i]. 
// Construct a binary search tree of all keys such that the total cost of all 
// the searches is as small as possible.
// Let us first define the cost of a BST. The cost of a BST node is the level 
// of that node multiplied by its frequency. The level of the root is 1.

// Examples:  

// Input:  keys[] = {10, 12}, freq[] = {34, 50}
// There can be following two possible BSTs 
//         10                       12
//           \                     / 
//            12                 10
//           I                     II
// Frequency of searches of 10 and 12 are 34 and 50 respectively.
// The cost of tree I is 34*1 + 50*2 = 134
// The cost of tree II is 50*1 + 34*2 = 118 


// Input:  keys[] = {10, 12, 20}, freq[] = {34, 8, 50}
// There can be following possible BSTs
//     10                12                 20         10              20
//       \             /    \              /             \            /
//       12          10     20           12               20         10  
//         \                            /                 /           \
//          20                        10                12             12  
//      I               II             III             IV             V
// Among all possible BSTs, cost of the fifth BST is minimum.  
// Cost of the fifth BST is 1*50 + 2*34 + 3*8 = 142 


int optimalSearchTree(int keys[], int freq[], int n){

    int cost[n][n];
 
    /* cost[i][j] = Optimal cost of binary search tree
    that can be formed from keys[i] to keys[j].
    cost[0][n-1] will store the resultant cost */
 
    // For a single key, cost is equal to frequency of the key
    for (int i = 0; i < n; i++)
        cost[i][i] = freq[i];
 
    // Now we need to consider chains of length 2, 3, ... .
    // L is chain length.
    for (int L = 2; L <= n; L++)
    {
        // i is row number in cost[][]
        for (int i = 0; i <= n-L+1; i++)
        {
            // Get column number j from row number i and chain length L
            int j = i+L-1;
            cost[i][j] = INT_MAX;
            int sm = sum(freq, i, j);
 			// int cost = optCost(freq, i, r - 1) + optCost(freq, r + 1, j);
	   //      mn = min(mn,cost);
            // Try making all keys in interval keys[i..j] as root
            for (int r = i; r <= j; r++)
            {
	            // c = cost when keys[r] becomes root of this subtree
	            int c = ((r > i)? cost[i][r-1]:0)+((r < j)? cost[r+1][j]:0) + sm;
	            									
	            cost[i][j] = min(c,cost[i][j]);
            }
        }
    }
    return cost[0][n-1]; //mn+off_set_sum
}
 
// A utility function to get sum of array elements
// freq[i] to freq[j]
int sum(int freq[], int i, int j)
{
    int s = 0;
    for (int k = i; k <= j; k++)
    s += freq[k];
    return s;
}