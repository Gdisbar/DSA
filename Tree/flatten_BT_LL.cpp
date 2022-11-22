114. Flatten Binary Tree to Linked List
==========================================
Given the root of a binary tree, flatten the tree into a "linked list":

    The "linked list" should use the same TreeNode class where the 
    right child pointer points to the next node in the list and the left 
    child pointer is always null.
    The "linked list" should be in the same order as a pre-order traversal 
    of the binary tree.

 							 1            1 
 						   /   \           \
 						  2     5     =>    2
 						 / \     \           \
 						3   4     6           3
                                               \
                                                4
                                                 \
                                                  5
                                                   \
                                                    6

Example 1:

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

//TC : n , SC : n

class Solution {
public:
    TreeNode* prev=nullptr;
    void flatten(TreeNode* root) {
          if (!root) return;
          flatten(root->right);
          flatten(root->left);
          root->right = prev;
          root->left = nullptr;
          prev = root;
    }
};

// Morris-Traversal , TC : n , SC : 1


class Solution {
  TreeNode * prev = NULL;
  public:
   void flatten(TreeNode* root) {
        TreeNode* cur = root;
		while (cur)
		{
			if(cur->left)
			{
				TreeNode* pre = cur->left;
				while(pre->right)
				{
					pre = pre->right;
				}
				pre->right = cur->right;
				cur->right = cur->left;
				cur->left = NULL;
			}
			cur = cur->right;
		}
    }

