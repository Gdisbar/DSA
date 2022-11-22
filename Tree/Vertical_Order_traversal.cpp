987. Vertical Order Traversal of a Binary Tree
=================================================
Given the root of a binary tree, calculate the vertical order traversal 
of the binary tree.

For each node at position (row, col), its left and right children will be 
at positions (row + 1, col - 1) and (row + 1, col + 1) respectively. 
The root of the tree is at (0, 0).

The vertical order traversal of a binary tree is a list of top-to-bottom 
orderings for each column index starting from the leftmost column and ending 
on the rightmost column. There may be multiple nodes in the same row and same 
column. In such a case, sort these nodes by their values.

Return the vertical order traversal of the binary tree.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation:
Column -1: Only node 9 is in this column.
Column 0: Nodes 3 and 15 are in this column in that order from top to bottom.
Column 1: Only node 20 is in this column.
Column 2: Only node 7 is in this column.

Example 2:

Input: root = [1,2,3,4,5,6,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
Column -2: Only node 4 is in this column.
Column -1: Only node 2 is in this column.
Column 0: Nodes 1, 5, and 6 are in this column.
          1 is at the top, so it comes first.
          5 and 6 are at the same position (2, 0), so we order them by their value, 5 before 6.
Column 1: Only node 3 is in this column.
Column 2: Only node 7 is in this column.



// 69% time, 33% space
vector<vector<int>> verticalTraversal(TreeNode* root) {
        if(root==nullptr) return vector<vector<int>> {};
        map<int,map<int,multiset<int>>> nodes;
        queue<pair<TreeNode*,pair<int,int>>> q;
        q.push(make_pair(root,make_pair(0,0)));
        pair<TreeNode*,pair<int,int>> p;
        TreeNode* node;
        int x,y;
        while(!q.empty()){
            p=q.front();
            q.pop();
            node=p.first;
            x=p.second.first;
            y=p.second.second;
            nodes[x][y].insert(node->val);
            if(node->left) q.push(make_pair(node->left,make_pair(x-1,y+1)));
            if(node->right) q.push(make_pair(node->right,make_pair(x+1,y+1)));
        }
        vector<vector<int>> ans;
        for(auto p : nodes){
            vector<int> col;
            for(auto q : p.second){
                col.insert(col.end(),q.second.begin(),q.second.end());
            }
            ans.push_back(col);
        }
        return ans;
    }

// using DFS

class Solution {
public:
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        map<int, map<int, set<int>>> nodes;
        traverse(root, 0, 0, nodes);
        vector<vector<int>> ans;
        for (auto p : nodes) {
            vector<int> col;
            for (auto q : p.second) {
                col.insert(col.end(), q.second.begin(), q.second.end());
            }
            ans.push_back(col);
        }
        return ans;
    }
private:
    void traverse(TreeNode* root, int x, int y, map<int, map<int, set<int>>>& nodes) {
        if (root) {
            nodes[x][y].insert(root -> val);
            traverse(root -> left, x - 1, y + 1, nodes);
            traverse(root -> right, x + 1, y + 1, nodes);
        }
    }
};