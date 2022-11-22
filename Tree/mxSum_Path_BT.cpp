124. Binary Tree Maximum Path Sum
====================================
// A path in a binary tree is a sequence of nodes where each pair of adjacent 
// nodes in the sequence has an edge connecting them. A node can only appear in 
// the sequence at most once. Note that the path does not need to pass through 
// the root.

// The path sum of a path is the sum of the node''s values in the path.

// Given the root of a binary tree, return the maximum path sum of any 
// non-empty path.

 

// Example 1:

// Input: root = [1,2,3]
// Output: 6
// Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

// Example 2:

// Input: root = [-10,9,20,null,null,15,7]
// Output: 42
// Explanation: The optimal path is 15 -> 20 -> 7 with a path sum 
// of 15 + 20 + 7 = 42.

//TC : n , SC : 1 (auxiliary stack space n)

class Solution {
private:
    int mx=INT_MIN;
    int preOrder(TreeNode* root){
        if(!root) return 0;
        int leftmx=max(0,preOrder(root->left));
        int rightmx=max(0,preOrder(root->right));
        int curmx=root->val+leftmx+rightmx;
        mx=max(mx,curmx);
        return root->val+max(leftmx,rightmx);
    }
public:
    int maxPathSum(TreeNode* root) {
        preOrder(root);
        return mx;
    }
};

//faster implementation

int solve(TreeNode* root,int &res)
    {
        // Base Case 
        if(root==NULL) return 0;
        int ls = solve(root->left,res);
        int rs = solve(root->right,res);
        int temp = max(max(ls,rs)+root->val,root->val);
        int ans = max(temp,ls+rs+root->val);
        res = max(res,ans);
        return temp;
    }