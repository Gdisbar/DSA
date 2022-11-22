102. Binary Tree Level Order Traversal
===========================================
// Given the root of a binary tree, return the level order traversal of its 
// nodes'' values. (i.e., from left to right, level by level).

 

// Example 1:

// Input: root = [3,9,20,null,null,15,7]
// Output: [[3],[9,20],[15,7]]

// Example 2:

// Input: root = [1]
// Output: [[1]]

// Example 3:

// Input: root = []
// Output: []

// 15% faster, 85% less memory ---> 88% faster, 15% less memory

vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if(!root) return ans;
        queue<TreeNode*> q;
        // TreeNode* tmp;
        // int sz;
        q.push(root);
        while(!q.empty()){
            int sz=q.size(); // declare outside of loop
            vector<int> res;
            //res.clear();
            for(int i=0;i<sz;++i){
                auto tmp=q.front(); //declare outside of loop
                q.pop();
                res.push_back(tmp->val);
                if(tmp->left) q.push(tmp->left);
                if(tmp->right) q.push(tmp->right);
            }
            ans.push_back(res);
        }
        return ans;
    }


// Python

// # Definition for a binary tree node.
// # class TreeNode:
// #     def __init__(self, val=0, left=None, right=None):
// #         self.val = val
// #         self.left = left
// #         self.right = right
// class Solution:
//     def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
//         if not root:
//             return []
//         ans , q = [] , [root]
//         while q:
//             ans.append([node.val for node in q])
//             tmp=[]
//             for node in q:
//                 tmp.extend([node.left,node.right])
//             q=[leaf for leaf in tmp if leaf]
        
//         return ans
 


Reverse Level Order Traversal       
===============================
//Time Complexity: The worst-case time complexity of this method is O(n^2). 
//For a skewed tree, printGivenLevel() takes O(n) time where n is the number 
//of nodes in the skewed tree. So time complexity of printLevelOrder() 
//is O(n) + O(n-1) + O(n-2) + .. + O(1) which is O(n^2).


//DFS

void reverseLevelOrder(node* root)
{
    int h = height(root);
    int i;
    for (i=h; i>=1; i--) //THE ONLY LINE DIFFERENT FROM NORMAL LEVEL ORDER
        printGivenLevel(root, i);
}
 
/* Print nodes at a given level */
void printGivenLevel(node* root, int level)
{
    if (root == NULL)
        return;
    if (level == 1)
        cout << root->data << " ";
    else if (level > 1)
    {
        printGivenLevel(root->left, level - 1);
        printGivenLevel(root->right, level - 1);
    }
}
 
/* Compute the "height" of a tree -- the number of
    nodes along the longest path from the root node
    down to the farthest leaf node.*/
int height(node* node)
{
    if (node == NULL)
        return 0;
    else
    {
        /* compute the height of each subtree */
        int lheight = height(node->left);
        int rheight = height(node->right);
 
        /* use the larger one */
        if (lheight > rheight)
            return(lheight + 1);
        else return(rheight + 1);
    }
}


// BFS

void reverseLevelOrder(node* root)
{
    stack <node *> S;
    queue <node *> Q;
    Q.push(root);
 
    // Do something like normal level order traversal order. Following are the
    // differences with normal level order traversal
    // 1) Instead of printing a node, we push the node to stack
    // 2) Right subtree is visited before left subtree
    while (Q.empty() == false)
    {
        /* Dequeue node and make it root */
        root = Q.front();
        Q.pop();
        S.push(root);
 
        /* Enqueue right child */
        if (root->right)
            Q.push(root->right); // NOTE: RIGHT CHILD IS ENQUEUED BEFORE LEFT
 
        /* Enqueue left child */
        if (root->left)
            Q.push(root->left);
    }
 
    // Now pop all items from stack one by one and print them
    while (S.empty() == false)
    {
        root = S.top();
        cout << root->data << " ";
        S.pop();
    }
}