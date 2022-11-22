968. Binary Tree Cameras
=============================
// You are given the root of a binary tree. We install cameras on the tree nodes 
// where each camera at a node can monitor its parent, itself, and its 
// immediate children.

// Return the minimum number of cameras needed to monitor all nodes of the tree.
// 								0
// 							   /
// 							  C  
// 							 / \
// 							0 	0
 

// Example 1:

// Input: root = [0,0,null,0,0]
// Output: 1
// Explanation: One camera is enough to monitor all nodes if placed as shown.

// 								0
// 							   /
// 							  C  
// 							 / 
// 							0 
// 						   /
// 					      C
// 					      \
// 					       0 

// Example 2:

// Input: root = [0,0,null,0,null,0,null,null,0]
// Output: 2
// Explanation: At least two cameras are needed to monitor all nodes of the 
// tree. The above image shows one of the valid configurations of camera 
// placement.

 

// Intuition:

// Consider a node in the tree.It can be covered by its parent, itself, 
// its two children.Four options.

// Consider the root of the tree.It can be covered by left child, or right 
// child, or itself.Three options.

// Consider one leaf of the tree.It can be covered by its parent or by itself.
// Two options.

// If we set a camera at the leaf, the camera can cover the leaf and its parent.
// If we set a camera at its parent, the camera can cover the leaf, its 
// parent and its sibling.

// We can see that the second plan is always better than the first.
// Now we have only one option, set up camera to all leaves'' parent.

// Here is our greedy solution:

//     Set cameras on all leaves'' parents, then remove all covered nodes.
//     Repeat step 1 until all nodes are covered.

// Explanation:

// Apply a recusion function dfs.
// Return 0 if it''s a leaf.
// Return 1 if it''s a parent of a leaf, with a camera on this node.
// Return 2 if it''s coverd, without a camera on this node.

// For each node,
// if it has a child, which is leaf (node 0), then it needs camera.
// if it has a child, which is the parent of a leaf (node 1), then it''s covered.

// If it needs camera, then res++ and we return 1.
// If it''s covered, we return 2.
// Otherwise, we return 0.


    // int res = 0;
    // int minCameraCover(TreeNode* root) {
    //     return (dfs(root) < 1 ? 1 : 0) + res;
    // }

    // int dfs(TreeNode* root) {
    //     if (!root) return 2;
    //     int left = dfs(root->left), right = dfs(root->right);
    //     if (left == 0 || right == 0) {
    //         res++;
    //         return 1;
    //     }
    //     return left == 1 || right == 1 ? 2 : 0;


class Solution
{
private:
    int needToCover = -1, covered = 0, count = 0, hasCamera = 1;
    int solve(TreeNode *root)
    {
        if (root == NULL)
            return covered;

        int l = solve(root->left);
        int r = solve(root->right);

        if (l == needToCover || r == needToCover)
        {
            count++;
            return hasCamera;
        }
        else if (l == hasCamera || r == hasCamera)
            return covered;
        else
            return needToCover;
    }
public:
    int minCameraCover(TreeNode *root)
    {
        count = 0;
        if (solve(root) == needToCover)
            count += 1;
        return count;
    }
};



//                                       0   (root,C++)       
//                                      / 
// 									0 
// 								   /
// 								  0 (C)
// 								 / \
// 								0   0
// 							   /
// 							  0 (C)
// 							 /
// 							0 
// It is also needed for some other trees like [0,0,null,0,null,0,0,0,null,0,null]. 
// I think whenever a node is in NO_CAMERA state, then we need to monitor it by putting 
// a camera in the parent. But the root has no parent. So, if root gets in NO_CAMERA 
// state at the end, we need to place a camera at the root itself to cover it.