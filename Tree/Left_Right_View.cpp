199. Binary Tree Right Side View
=====================================
// Given the root of a binary tree, imagine yourself standing on the right side 
// of it, return the values of the nodes you can see ordered from top to bottom.

 

// Example 1:

// Input: root = [1,2,3,null,5,null,4]
// Output: [1,3,4]

// Example 2:

// Input: root = [1,null,3]
// Output: [1,3]

// Example 3:

// Input: root = []
// Output: []

//DFS

class Solution {
private:
    void helper(TreeNode* root,int level,vector<int> &res){
        if(!root) return;
        if(res.size()==level) res.push_back(root->val);
        helper(root->right,level+1,res);
        helper(root->left,level+1,res);
    }
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        helper(root,0,res);
        return res;
    }
};

//BFS ,faster
vector<int> rightSideView(TreeNode* root) {
        if (!root) {
            return {};
        }
        vector<int> view;
        queue<TreeNode*> todo;
        todo.push(root);
        while (!todo.empty()) {
            int n = todo.size();
            for (int i = 0; i < n; i++) {
                TreeNode* node = todo.front();
                todo.pop();
                if (i == n - 1) {
                    view.push_back(node -> val);
                }
                if (node -> left) {
                    todo.push(node -> left);
                }
                if (node -> right) {
                    todo.push(node -> right);
                }
            }
        }
        return view;
    }

Left View of Binary Tree 
============================
// Given a Binary Tree, print Left view of it. Left view of a Binary Tree 
// is set of nodes visible when tree is visited from Left side. The task is 
// to complete the function leftView(), which accepts root of the tree as argument.

// Left view of following tree is 1 2 4 8.

//           1
//        /     \
//      2        3
//    /  \     /   \
//   4    5   6    7
//    \
//     8   

// Example 1:

// Input:
//    1
//  /  \
// 3    2
// Output: 1 3

vector<int> leftView(Node *root)
{
   // Your code here
   if (!root) {
            return {};
        }
        vector<int> view;
        queue<Node*> todo;
        todo.push(root);
        while (!todo.empty()) {
            int n = todo.size();
            for (int i = 0; i < n; i++) {
                Node* node = todo.front();
                todo.pop();
                if (i == 0) {
                    view.push_back(node -> data);
                }
                if (node -> left) {
                    todo.push(node -> left);
                }
                if (node -> right) {
                    todo.push(node -> right);
                }
            }
        }
        return view;
}