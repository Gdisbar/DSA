1372. Longest ZigZag Path in a Binary Tree
============================================
// You are given the root of a binary tree.

// A ZigZag path for a binary tree is defined as follow:

//     Choose any node in the binary tree and a direction (right or left).
//     If the current direction is right, move to the right child of the current 
//     node; otherwise, move to the left child.
//     Change the direction from right to left or from left to right.
//     Repeat the second and third steps until you can't move in the tree.

// Zigzag length is defined as the number of nodes visited - 1. 
// (A single node has a length of 0).

// Return the longest ZigZag path contained in that tree.

 

// Example 1:

// Input: root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1,null,1]
// Output: 3
// Explanation: Longest ZigZag path in blue nodes (right -> left -> right).

// Example 2:
					 //     1
					 // l /  \
					 //  1    1
					 // r \ 
					 //    1 
					 // l / \
					 //  1   1
					 // r \
					 //    1      	

// Input: root = [1,1,1,null,1,null,null,1,1,null,1]
// Output: 4
// Explanation: Longest ZigZag path in blue nodes (left -> right -> left -> right).

// Example 3:

// Input: root = [1]
// Output: 0

// Explanation

// Recursive return [left, right, result], where:
// left is the maximum length in direction of root.left
// right is the maximum length in direction of root.right
// result is the maximum length in the whole sub tree.

// Complexity

// Time O(N)
// Space O(height)

    int longestZigZag(TreeNode* root) {
        return dfs(root)[2];
    }

    vector<int> dfs(TreeNode* root) {
        if (!root) return { -1, -1, -1};
        vector<int> left = dfs(root->left), right = dfs(root->right);
        int res = max(max(left[1], right[0]) + 1, max(left[2], right[2]));
        return {left[1] + 1, right[0] + 1, res};
    }



    def longestZigZag(self, root):
        def dfs(root):
            if not root: return [-1, -1, -1]
            left, right = dfs(root.left), dfs(root.right)
            return [left[1] + 1, right[0] + 1, max(left[1] + 1, right[0] + 1, left[2], right[2])]
        return dfs(root)[-1]
