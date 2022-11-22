98. Validate Binary Search Tree
=================================
// Given the root of a binary tree, determine if it is a valid binary search 
// tree (BST).

// A valid BST is defined as follows:

//     The left subtree of a node contains only nodes with keys less than the node's 
//     key.
//     The right subtree of a node contains only nodes with keys greater than the 
//     node's key.
//     Both the left and right subtrees must also be binary search trees.

 

// Example 1:

// Input: root = [2,1,3]
// Output: true

// Example 2:

// Input: root = [5,1,4,null,null,3,6]
// Output: false
// Explanation: The root node's value is 5 but its right child's value is 4.




// do an inorder  + check if inorder is sorted in ascending order

// using long is risky if root->val is also long

bool isValidBST(TreeNode* root, long min = LONG_MIN, long max = LONG_MAX){ 
       if(root == NULL)
           return true;
       if((root->val >= max) || (root->val <= min))
           return false;
       else
           return isValidBST(root->left,min,root->val) &&  
       				isValidBST(root->right,root->val,max);
   }

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid_bst(root, min_val, max_val):
            if root is None:
                return True
            if root.val <= min_val or root.val >= max_val:
                return False
            return valid_bst(root.left, min_val, root.val) and valid_bst(root.right, root.val, max_val)
        return valid_bst(root, -2**31-1, 2**31)



bool IsBST(TreeNode* root, TreeNode* min, TreeNode* max){
	if(root==NULL) return true;
	if(min!=NULL && min->val>=root->val) return false;
	if(max!=NULL && max->val<=root->val) return false;

    bool LeftValid = IsBST(root->left,min,root);
    bool RightValid = IsBST(root->right,root,max);
    
    return LeftValid && RightValid; 
}
bool isValidBST(TreeNode* root) {
    return IsBST(root,NULL,NULL);
}

//Question : Binary Tree Inorder Traversal


public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    if(root == null) return list;
    Stack<TreeNode> stack = new Stack<>();
    while(root != null || !stack.empty()){
        while(root != null){
            stack.push(root);
            root = root.left;
        }
        root = stack.pop();
        list.add(root.val);
        root = root.right;
        
    }
    return list;
}

// Now, we can use this structure to find the Kth smallest element in BST.

// Question : Kth Smallest Element in a BST

 public int kthSmallest(TreeNode root, int k) {
     Stack<TreeNode> stack = new Stack<>();
     while(root != null || !stack.isEmpty()) {
         while(root != null) {
             stack.push(root);    
             root = root.left;   
         } 
         root = stack.pop();
         if(--k == 0) break;
         root = root.right;
     }
     return root.val;
 }

// We can also use this structure to solve BST validation question.

// Question : Validate Binary Search Tree

public boolean isValidBST(TreeNode root) {
   if (root == null) return true;
   Stack<TreeNode> stack = new Stack<>();
   TreeNode pre = null;
   while (root != null || !stack.isEmpty()) {
      while (root != null) {
         stack.push(root);
         root = root.left;
      }
      root = stack.pop();
      if(pre != null && root.val <= pre.val) return false;
      pre = root;
      root = root.right;
   }
   return true;
}

// (Below, duplicate values must be on the left side of a node. 
// If it needs to be on right side, then just switch the > and >= signs.)

class Solution {
   public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode prev = null;
        boolean onRightSideOfPrev = false;
        while(root != null || !stack.isEmpty()) {
            while(root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if(prev != null && ((!onRightSideOfPrev && prev.val > root.val) || 
            		(onRightSideOfPrev && prev.val >= root.val))) {
                return false;
            }
            prev = root;
            root = root.right;
            onRightSideOfPrev = root == null ? false : true;
        }
        return true;
    }
}

// Recursive

class Solution {
    TreeNode prev;
        
    public boolean isValidBST(TreeNode root) {
        if (root == null)
            return true;
        
        if(!isValidBST(root.left))
            return false;
        
        if (prev != null && prev.val >= root.val)
            return false;
        
        prev = root;
        
        if (!isValidBST(root.right))
            return false;
        
        return true;
        
        
    }
}