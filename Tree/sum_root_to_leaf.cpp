129. Sum Root to Leaf Numbers
================================
You are given the root of a binary tree containing digits from 0 to 9 only.

Each root-to-leaf path in the tree represents a number.

    For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.

Return the total sum of all root-to-leaf numbers. Test cases are generated 
so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.

 

Example 1:

Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

Example 2:

Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.

int sumNumbers(TreeNode* root) {
        return dfs(root, 0);
    }
    // preorder dfs traversal
    int dfs(TreeNode* root, int cur) {
        if(!root) return 0;
        cur = cur * 10 + root -> val;   // append current node's digit
        // leaf node - return current number to be added to total sum
        if(!root -> left && !root -> right)   
            return cur;
        return dfs(root -> left, cur) + dfs(root -> right, cur);   
        // recurse for child if current node is not leaf
    }  


// Time Complexity : O(N), where N is the number of nodes in the tree. 
// We are doing a standard DFS traversal which takes O(N) time
// Space Complexity : O(H), where H is the maximum depth of tree. This space is 
// required for implicit recursive stack space. In the worst case, 
// the tree maybe skewed and H = N in which case space required is equal to O(N).



int sumNumbers(TreeNode* root) {
        int sum = 0, cur = 0, depth = 0;
        while(root) {
            if(root -> left) {
                auto pre = root -> left;
                depth = 1;
                while(pre -> right && pre -> right != root) 
                    pre = pre -> right, depth++;
                if(!pre -> right) {
                    pre -> right = root;
                    cur = cur * 10 + root -> val;
                    root = root -> left;
                } else {
                    pre -> right = nullptr;
                    if(!pre -> left) sum += cur;
                    cur /= pow(10, depth);
                    root = root -> right;
                }
            } else {
                cur = cur * 10 + root -> val;
                if(!root -> right) sum += cur;
                root = root -> right;
            }
        }
        return sum;
    }

// Time Complexity : O(N)
// Space Complexity : O(1)