450. Delete Node in a BST
============================
// Given a root node reference of a BST and a key, delete the node with the given 
// key in the BST. Return the root node reference (possibly updated) of the BST.

// Basically, the deletion can be divided into two stages:

//     Search for a node to remove.
//     If the node is found, delete the node.

 

// Example 1:

// Input: root = [5,3,6,2,4,null,7], key = 3
// Output: [5,4,6,2,null,null,7]
// Explanation: Given key to delete is 3. So we find the node with value 3 and delete it.
// One valid answer is [5,4,6,2,null,null,7], shown in the above BST.
// Please notice that another valid answer is [5,2,6,null,4,null,7] and it''s also 
// accepted.

// Example 2:

// Input: root = [5,3,6,2,4,null,7], key = 0
// Output: [5,3,6,2,4,null,7]
// Explanation: The tree does not contain a node with value = 0.

// Example 3:

// Input: root = [], key = 0
// Output: []

 

// Constraints:

//     The number of nodes in the tree is in the range [0, 104].
//     -105 <= Node.val <= 105
//     Each node has a unique value.
//     root is a valid binary search tree.
//     -10^5 <= key <= 10^5

TreeNode* deleteNode(TreeNode* root, int key) {
    if(!root) return nullptr;
    //We frecursively call the function until we find the target node
    if(key < root->val) root->left = deleteNode(root->left, key);     
    else if(key > root->val) root->right = deleteNode(root->right, key);       
    else{
    	//No child condition
        if(!root->left && !root->right) return NULL;  
        //One child contion -> replace the node with it's child        
        if (!root->left || !root->right)
            return root->left ? root->left : root->right;    
		//Two child condition 
		//(or) TreeNode *temp = root->right;  
		// while(temp->left != NULL) temp = temp->left;
		// root->val = temp->val;
		// root->right = deleteNode(root->right, temp);	

        TreeNode* temp = root->left;     
        //largest in left subtree                   
        while(temp->right != NULL) temp = temp->right;     
        root->val = temp->val;                            
        root->left = deleteNode(root->left, temp->val);  	
    }
    return root;
}

class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        
        # base case
        if not root: return None
        
        # base case (almost) with key found
        if root.val == key:
            # 1.a leaf or single child
            if not root.right: return root.left
            
            # 1.b leaf or single child
            if not root.left: return root.right
            
            # 2: both child node exist
            if root.left and root.right:
                # 2.a.1: start with right node of deleted node
                temp = root.right
                
                # 2.a.2: find minimum node in left subtree
                # we are going to replace minimum in left subtree with value at root
                while temp.left: 
                    temp = temp.left
                
                # 2.b: replace value with minimum value in right subtree
                root.val = temp.val
                
                # 2.c: ** key step ** recurse on root.right but with key  = root.val (min val in right subtree)
                root.right = self.deleteNode(root.right, root.val)
        
        # recursion steps
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
            
        return root
