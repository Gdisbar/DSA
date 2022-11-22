105. Construct Binary Tree from Preorder and Inorder Traversal
================================================================
Given two integer arrays preorder and inorder where preorder is the 
preorder traversal of a binary tree and inorder is the inorder traversal 
of the same tree, construct and return the binary tree.


Example 1:

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return dfs(preorder, inorder, nullptr);
    }
    
    TreeNode* dfs(vector<int> &pre, vector<int> &in, TreeNode *parent) {
        if (pre.empty() || parent && in[0] == parent->val)
            return nullptr;
        
        TreeNode *curr = new TreeNode(pre[0]);
        pre.erase(pre.begin());
        curr->left = dfs(pre, in, curr);
        in.erase(in.begin());
        curr->right = dfs(pre, in, parent);
        return curr;
    }


    

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # Preorder Traversal: Root -> Left -> Right
        # Inorder Traversal: Left -> Root -> Right
        # Hence, at any step, first value in the preorder traversal represents
        # root of the binary tree at that point.
        # If that value is at the i'th index in inorder list,
        # then all the values to its left are part of left subtree.
        # Similarly, all the values to its right are part of the right subtree.
        if(len(inorder) == 0):
            return None
        val = preorder.pop(0)
        node = TreeNode(val)
        idx = inorder.index(val)
        leftInorder = inorder[:idx]
        rightInorder = inorder[(idx + 1):]
        # Since we traverse left node before right node in preorder tranversal,
        # node.left needs to be evaluated before node.right
        node.left = self.buildTree(preorder, leftInorder)
        node.right = self.buildTree(preorder, rightInorder)
        return node
