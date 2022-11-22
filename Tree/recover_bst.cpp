99. Recover Binary Search Tree
===================================
// You are given the root of a binary search tree (BST), 
// where the values of exactly two nodes of the tree were swapped by mistake. 
// Recover the tree without changing its structure.

 

// Example 1:

// Input: root = [1,3,null,null,2]
// Output: [3,1,null,null,2]
// Explanation: 3 cannot be a left child of 1 because 3 > 1. 
// Swapping 1 and 3 makes the BST valid.

// Example 2:

// Input: root = [3,1,4,null,null,2]
// Output: [2,1,4,null,null,3]
// Explanation: 2 cannot be in the right subtree of 3 because 2 < 3. 
// Swapping 2 and 3 makes the BST valid.

// Brute force : store inorder + sort them + again do another inorder & swap 

class Solution {
private:
    TreeNode* first;
    TreeNode* prev;
    TreeNode* middle;
    TreeNode* last; 
    void inOrder(TreeNode* root){
        if(!root) return;
        inOrder(root->left);
        // for sorted order root/curr > prev
        if(prev!=nullptr&&(root->val<prev->val)){  
            if(first==nullptr){     .             //     f  m              l
                first=prev;		  //first violation , 3 |25|7| 8 10 15 20 |5|
                middle=root;  //if no 1st violation found the swap(f,m) i.e 2nd violation
            }
            else{                   //               f m
               last=root;  //second violation , 3 5 |8|7| 10 15 20 25
            }
        }
        //this step storing the previous value
        prev=root;
        inOrder(root->right);
    }
   
public:
    void recoverTree(TreeNode* root) {
        first=middle=last=nullptr; 
        prev=new TreeNode(INT_MIN); 
        inOrder(root);
        if(first&&last) swap(first->val,last->val); //non adjacent pair
        else if(first&&middle) swap(first->val,middle->val); //adjacent pair
        
    }
};