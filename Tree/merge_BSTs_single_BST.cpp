1932. Merge BSTs to Create Single BST
=======================================
// You are given n BST (binary search tree) root nodes for n separate BSTs stored 
// in an array trees (0-indexed). Each BST in trees has at most 3 nodes, and no two 
// roots have the same value. In one operation, you can:

//     Select two distinct indices i and j such that the value stored at one of the 
//     leaves of trees[i] is equal to the root value of trees[j].
//     Replace the leaf node in trees[i] with trees[j].
//     Remove trees[j] from trees.

// Return the root of the resulting BST if it is possible to form a valid BST after 
// performing n - 1 operations, or null if it is impossible to create a valid BST.

// A BST (binary search tree) is a binary tree where each node satisfies the following 
// property:

//     Every node in the node's left subtree has a value strictly less than the node's 
//     value.
//     Every node in the node's right subtree has a value strictly greater than the 
//     node's value.

// A leaf is a node that has no children.

 						
//  						2      3        5
//  					   /      / \      /
//  					  1      2   5    4

//  					  			 3
//  					  		   /   \
//  					  		  2     5     
//  					  		 /
//  					  		1

//  					  			 3
//  					  		   /   \
//  					  		  2     5    
//  					  		 /     /
//  					  		1     4
// Example 1:

// Input: trees = [[2,1],[3,2,5],[5,4]]
// Output: [3,2,5,1,null,4]
// Explanation:
// In the first operation, pick i=1 and j=0, and merge trees[0] into trees[1].
// Delete trees[0], so trees = [[3,2,5,1],[5,4]].

// In the second operation, pick i=0 and j=1, and merge trees[1] into trees[0].
// Delete trees[1], so trees = [[3,2,5,1,null,4]].

// The resulting tree, shown above, is a valid BST, so return its root.

// Example 2:

// Input: trees = [[5,3,8],[3,2,6]]
// Output: []
// Explanation:
// Pick i=0 and j=1 and merge trees[1] into trees[0].
// Delete trees[1], so trees = [[5,3,8,2,6]].

// The resulting tree is shown above. This is the only valid operation that can be performed, but the resulting tree is not a valid BST, so return null.

// Example 3:

// Input: trees = [[5,4],[3]]
// Output: []
// Explanation: It is impossible to perform any operations.

 

// Constraints:

//     n == trees.length
//     1 <= n <= 5 * 104
//     The number of nodes in each tree is in the range [1, 3].
//     Each node in the input may have children but no grandchildren.
//     No two roots of trees have the same value.
//     All the trees in the input are valid BSTs.
//     1 <= TreeNode.val <= 5 * 10^4.

Approach 1: Build from root
=============================
// This solution identifies the topmost root, and then traverses from it, joining 
// leaves with matching roots.

//     Populate a hashmap {value: root} . All root values are guaranteed to be unique.
//     Count values among all trees.
//     Identify a root of the combined tree; it's value must be counted only once.
//     Traverse from the root:
//         Check BST validity, like in 98. Validate Binary Search Tree.
//         Join leaves with roots, matching leaf and root value using the map.
//     If the combined tree is valid, and it includes all roots - return the root of 
//     the combined tree.

bool traverse(TreeNode* root, unordered_map<int, TreeNode*> &m, 
	 			int min_left = INT_MIN, int max_right = INT_MAX) {
    if (root == nullptr) 
        return true;
    if (root->val <= min_left || root->val >= max_right) //check validity of BST
        return false;
    if (root->left == root->right) {
        auto it = m.find(root->val); //find the matching value in map , if it's not current root
        if (it != end(m) && root != it->second) {
            root->left = it->second->left; //add left subtree to current root
            root->right = it->second->right; //add right subtree to current right
            m.erase(it); // remove processed root from map
        }
    }
    return traverse(root->left, m, min_left, root->val) && traverse(root->right, m, root->val, max_right);
}    
TreeNode* canMerge(vector<TreeNode*>& trees) {
    unordered_map<int, TreeNode*> m;
    unordered_map<int, int> cnt;
    for (auto &t : trees) {
        m[t->val] = t; //m={value:root} , contains roots of all trees
        //count values among trees
        ++cnt[t->val];
        ++cnt[t->left ? t->left->val : 0];
        ++cnt[t->right ? t->right->val : 0];
    }
    for (auto &t : trees)
        if (cnt[t->val] == 1) //identify root of combined tree
            return traverse(t, m) && m.size() == 1 ? t : nullptr;
    return nullptr;
}


// First of all, we want to find a root node to start the traversal from, and we 
// can do so by finding the node without any incoming edge (indeg = 0). 
// If there''s zero or more than one roots, we cannot create a single BST.

// To traverse through nodes, we need to go from one BST to another. We achieve 
// this with the help of a value-to-node map (nodes).

// There are also two edges cases we need to check:

//     There is no cycle
//     We traverse through all nodes


//     Time complexity: O(N)
//     Space complexity: O(N)


class Solution:
    def canMerge(self, trees: List[TreeNode]) -> TreeNode:
        nodes = {}
        indeg = collections.defaultdict(int)
        for t in trees:
            if t.val not in indeg:
                indeg[t.val] = 0
            if t.left:
                indeg[t.left.val] += 1
                if t.left.val not in nodes: nodes[t.left.val] = t.left
            if t.right:
                indeg[t.right.val] += 1
                if t.right.val not in nodes: nodes[t.right.val] = t.right
            nodes[t.val] = t
            
        # check single root
        sources = [k for k, v in indeg.items() if v == 0]
        if len(sources) != 1: return None
        
        self.cur = float('-inf')
        self.is_invalid = False
        seen = set()
        def inorder(val):
            # check cycle
            if val in seen:
                self.is_invalid = True
                return
            seen.add(val)
            node = nodes[val]
            if node.left: node.left = inorder(node.left.val)
            # check inorder increasing
            if val <= self.cur:
                self.is_invalid = True
                return
            self.cur = val
            if node.right: node.right = inorder(node.right.val)
            return node
        
        root = inorder(sources[0])
        # check full traversal
        if len(seen) != len(nodes) or self.is_invalid:
            return None
        return root


// # Definition for a binary tree node.
// # class TreeNode:
// #     def __init__(self, val=0, left=None, right=None):
// #         self.val = val
// #         self.left = left
// #         self.right = right
// class Solution:
//     def canMerge(self, trees: List[TreeNode]) -> Optional[TreeNode]:
//         nodes={}
//         indeg=collections.defaultdict(int)
//         for t in trees:
//             if t.val not in indeg:
//                 indeg[t.val]=0
//                 if t.left:
//                     indeg[t.left.val]+=1
//                     if t.left.val not in nodes:
//                         nodes[t.left.val]=t.left
//                 if t.right:
//                     indeg[t.right.val]+=1
//                     if t.right.val not in nodes:
//                         nodes[t.right.val]=t.right
                        
//                 nodes[t.val]=t
//         #check single root
//         sources=[k for k,v in indeg.items() if v==0]
//         if len(sources)!=1 : 
//             return None
        
//         self.cur=float('-inf')
//         self.is_invalid=False
//         visit=set()
//         def inorder(val):
//             if val in visit:
//                 self.is_invalid=True
//                 return
//             visit.add(val)
//             node=nodes[val]
//             if node.left:
//                 node.left=inorder(node.left.val)
//             if val<=self.cur:
//                 self.is_invalid=True
//                 return
//             self.cur=val
//             if node.right:
//                 node.right=inorder(node.right.val)
//             return node
        
//         root=inorder(sources[0])
//         #check full traversal
//         if len(visit)!=len(nodes) or self.is_invalid:
//             return None
//         return root