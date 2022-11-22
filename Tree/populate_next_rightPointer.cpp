116. Populating Next Right Pointers in Each Node
====================================================
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}

Populate each next pointer to point to its next right node. 
If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

 

Example 1:

Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), 
your function should populate each next pointer to point to its next right node, 
just like in Figure B. The serialized output is in level order as connected by 
the next pointers, with '#' signifying the end of each level.

Example 2:

Input: root = []
Output: []

//BFS --> 20% faster , 60% less memory

    Node* connect(Node* root) {
        if(!root) return nullptr;
        Node* tmp;
        int sz;
        queue<Node*> q;
        q.push(root);
        //root->next=nullptr;
        while(!q.empty()){
            Node* rightNode = nullptr;
            sz=q.size();
            for(int i=sz-1;i>=0;--i){
              tmp=q.front();
              q.pop();
              tmp->next=rightNode;
              rightNode=tmp;
              if(tmp->right) q.push(tmp->right);
              if(tmp->left) q.push(tmp->left);
            }
        }
        return root;
    }


// BFS + space optimized --> 705 less space

Node* connect(Node* root) {
        auto head = root;
        for(; root; root = root -> left) 
            for(auto cur = root; cur; cur = cur -> next)   // traverse each level - it's just BFS taking advantage of next pointers          
                if(cur -> left) {                          // update next pointers of children if they exist               
                    cur -> left -> next = cur -> right;
                    if(cur -> next) cur -> right -> next = cur -> next -> left;
                }
                else break;                                // if no children exist, stop iteration                                                  
        
        return head;
    }

// class Solution:
//     def connect(self, root):
//         head = root
//         while root:
//             cur, root = root, root.left
//             while cur:
//                 if cur.left:
//                     cur.left.next = cur.right
//                     if cur.next: cur.right.next = cur.next.left
//                 else: break
//                 cur = cur.next
                
//         return head

// DFS --> faster than BFS

Node* connect(Node* root) {
        if(!root) return nullptr;
        auto L = root -> left, R = root -> right, N = root -> next;
        if(L) {
            L -> next = R;                                // next of root's left is assigned as root's right
            if(N) R -> next = N -> left;                  // next of root's right is assigned as root's next's left (if root's next exist)
            connect(L);                                   // recurse left  - simple DFS 
            connect(R);                                   // recurse right
        }
        return root;
    }

// class Solution:
//     def connect(self, root):
//         if not root: return None
//         L, R, N = root.left, root.right, root.next
//         if L:
//             L.next = R
//             if N: R.next = N.left
//             self.connect(L)
//             self.connect(R)
//         return root