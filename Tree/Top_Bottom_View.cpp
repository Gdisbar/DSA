Top View of Binary Tree
===========================

Top view of a binary tree is the set of nodes visible when the tree is viewed from the top. Given a binary tree, print the top view of it. The output nodes can be printed in any order. Expected time complexity is O(n)

A node x is there in output if x is the topmost node at its horizontal distance. Horizontal distance of left child of a node x is equal to horizontal distance of x minus 1, and that of right child is horizontal distance of x plus 1. 

Example :

       1
    /     \
   2       3
  /  \    / \
 4    5  6   7
Top view of the above binary tree is
4 2 1 3 7

        1
      /   \
    2       3
      \   
        4  
          \
            5
             \
               6
Top view of the above binary tree is
2 1 3 6

// function should print the topView of
// the binary tree
void topView(struct Node* root)
{
    // Base case
    if (root == NULL) {
        return;
    }
 
    // Take a temporary node
    Node* temp = NULL;
 
    // Queue to do BFS
    queue<pair<Node*, int> > q;
 
    // map to store node at each horizontal distance
    map<int, int> mp;
 
    q.push({ root, 0 });
 
    // BFS
    while (!q.empty()) {
 
        temp = q.front().first;
        int d = q.front().second;
        q.pop();
 
        // If any node is not at that horizontal distance
        // just insert that node in map and print it
        if (mp.find(d) == mp.end()) {
            cout << temp->data << " ";
            mp[d] = temp->data;
        }
 
        // Continue for left node
        if (temp->left) {
            q.push({ temp->left, d - 1 });
        }
 
        // Continue for right node
        if (temp->right) {
            q.push({ temp->right, d + 1 });
        }
    }
}

Bottom View of Binary Tree
============================
Given a binary tree, print the bottom view from left to right.
A node is included in bottom view if it can be seen when we look at the tree from bottom.

                      20
                    /    \
                  8       22
                /   \        \
              5      3       25
                    /   \      
                  10    14

For the above tree, the bottom view is 5 10 3 14 25.
If there are multiple bottom-most nodes for a horizontal distance from root, then print the later one in level traversal. For example, in the below diagram, 3 and 4 are both the bottommost nodes at horizontal distance 0, we need to print 4.

                      20
                    /    \
                  8       22
                /   \     /   \
              5      3 4     25
                     /    \      
                 10       14

For the above tree the output should be 5 10 4 14 25.
 

Example 1:

Input:
       1
     /   \
    3     2
Output: 3 1 2
Explanation:
First case represents a tree with 3 nodes
and 2 edges where root is 1, left child of
1 is 3 and right child of 1 is 2. Thus nodes of the binary tree will be
printed as such 3 1 2.


    vector <int> bottomView(Node *root) {
        // Your Code Here
        if (root == NULL) {
            return vector<int> {};
        }
        Node* temp = NULL;
        queue<pair<Node*, int> > q;
        map<int, int> mp;
        q.push({ root, 0 });
        while (!q.empty()) {
            temp = q.front().first;
            int d = q.front().second;
            q.pop();
            //    cout << temp->data << " ";
            mp[d] = temp->data;
            if (temp->left) {
                q.push({ temp->left, d - 1 });
            }
            if (temp->right) {
                q.push({ temp->right, d + 1 });
            }
        }
        vector<int> res;
        for(auto p : mp) res.push_back(p.second);
        return res;
    }