Image Multiplication 
=======================
You are given a binary tree. Your task is pretty straightforward. 
You have to find the sum of the product of each node and its mirror 
image (The mirror of a node is a node which exists at the mirror position 
of the node in opposite subtree at the root.). Don’t take into account a 
pair more than once. The root node is the mirror image of itself.

 

Example 1:

Input:
      4         
    /    \
   5      6
Output:
46
Explanation:
Sum = (4*4) + (5*6) = 46

Example 2:

Input:
                       1                 
                   /        \
                 3            2
                  /  \         /  \
              7     6       5    4
            /   \    \     /  \    \
          11    10    15  9    8    12
Output:
332
Explanation:
Sum = (1*1) + (3*2) + (7*4) + (6*5) + (11*12) + (15*9) = 332

 

Your Task:
You need to complete the function imgMultiply() that takes root as 
parameter and returns the required sum.The answer may be very large, 
compute the answer modulo 10^9 + 7.


Expected Time Complexity: O(Number of nodes).
Expected Auxiliary Space: O(Height of the Tree).

class Solution
{
    long long modulus = 1000000007;
    public:
    void printInorder(Node* rootL, Node* rootR, long long &ans)
    {
        // We are using 2 pointers for the nodes
        // which are mirror image of each other
        // If both child are NULL return
        if (!rootL || !rootR)
            return;
     
        // Since inorder traversal is required
        // First left, then root and then right
        ans+=(rootL->data)*(rootR->data)%modulus;
        printInorder(rootL->left, rootR->right, ans);
        printInorder(rootL->right, rootR->left, ans);
    }
    long long imgMultiply(Node *root)
    {
        long long ans = (root->data*root->data);
        printInorder(root->left, root->right, ans);
        return ans%modulus;
    }
};