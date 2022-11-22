222. Count Complete Tree Nodes
================================
Given the root of a complete binary tree, return the number of the nodes 
in the tree.

According to Wikipedia, every level, except possibly the last, 
is completely filled in a complete binary tree, and all nodes in the last 
level are as far left as possible. It can have between 1 and 2^h nodes inclusive 
at the last level h.

Design an algorithm that runs in less than O(n) time complexity.

 

Example 1:

Input: root = [1,2,3,4,5,6]
Output: 6

Example 2:

Input: root = []
Output: 0

Example 3:

Input: root = [1]
Output: 1

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
// The idea is to find whether a subtree is full binary tree or not.
// If it is then we can directly count the nodes, otherwise check recursively.

int countNodes(TreeNode* root) {

        if(!root) return 0;

        int hl=0, hr=0;

        TreeNode *l=root, *r=root;

        while(l) {hl++;l=l->left;}

        while(r) {hr++;r=r->right;}

        if(hl==hr) return (1<<hl)-1; //pow(2,hl)-1

        //subtree of a complete binary tree is also a complete binary tree
        return 1+countNodes(root->left)+countNodes(root->right);

    }

// Let n be the total number of the tree. It is likely that you will get a 
// child tree as a perfect binary tree and a non-perfect binary tree (T(n/2)) 
// at each level.

// T(n) = T(n/2) + c1 lgn
//        = T(n/4) + c1 lgn + c2 (lgn - 1)
//        = ...
//        = T(1) + c [lgn + (lgn-1) + (lgn-2) + ... + 1]
//        = O(lgn*lgn)   