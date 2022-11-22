1339. Maximum Product of Splitted Binary Tree
======================================================
Given the root of a binary tree, split the binary tree into two subtrees by removing one edge such that the product of the sums of the subtrees is maximized.

Return the maximum product of the sums of the two subtrees. Since the answer may be too large, return it modulo 109 + 7.

Note that you need to maximize the answer before taking the mod and not after taking it.

 

Example 1:

Input: root = [1,2,3,4,5,6]
Output: 110
Explanation: Remove the red edge and get 2 binary trees with sum 11 and 10. Their product is 110 (11*10)

Example 2:

Input: root = [1,null,2,3,4,null,null,5,6]
Output: 90
Explanation: Remove the red edge and get 2 binary trees with sum 15 and 6.Their product is 90 (15*6)


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    long total=0;
    long mxprod=0;
    void preorder(TreeNode* root){
        if(!root) return;
        total+=(long)root->val;
        preorder(root->left);
        preorder(root->right);
    }
   long findprod(TreeNode* root){
       if(!root) return 0;
        long lt=findprod(root->left);
        long rt=findprod(root->right);
        long subtree=root->val+lt+rt;
        mxprod=max(mxprod,subtree*(total-subtree));
        return subtree;
    }
public:
    int maxProduct(TreeNode* root) {
        preorder(root);
        findprod(root);
        return (int)(mxprod%1000000007);
    }
};

// Time O(N)
// Space O(height)


class Solution {
public:
    long res = 0, total = 0, sub;
    int maxProduct(TreeNode* root) {
        total = s(root), s(root);
        return res % (int)(1e9 + 7);
    }

    int s(TreeNode* root) {
        if (!root) return 0;
        sub = root->val + s(root->left) + s(root->right);
        res = max(res, sub * (total - sub));
        return sub;
    }
};