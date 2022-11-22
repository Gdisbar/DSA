1145. Binary Tree Coloring Game
====================================
// Two players play a turn based game on a binary tree. We are given the root of 
// this binary tree, and the number of nodes n in the tree. n is odd, and each node 
// has a distinct value from 1 to n.

// Initially, the first player names a value x with 1 <= x <= n, and the second 
// player names a value y with 1 <= y <= n and y != x. The first player colors the 
// node with value x red, and the second player colors the node with value y blue.

// Then, the players take turns starting with the first player. In each turn, that 
// player chooses a node of their color (red if player 1, blue if player 2) and 
// colors an uncolored neighbor of the chosen node (either the left child, right 
// child, or parent of the chosen node.)

// If (and only if) a player cannot choose such a node in this way, they must pass 
// their turn. If both players pass their turn, the game ends, and the winner is 
// the player that colored more nodes.

// You are the second player. If it is possible to choose such a y to ensure you win 
// the game, return true. If it is not possible, return false.

//  									1
//  							     /    \
//  						    (B) 2      3  (R)
//  					          /   \   /  \
//  					         4     5 6    7
//  					       /  \  /  \
//  					      8   9 10  11


// if (B) color root=1, mx=8 
// // [4,5,8,9,10,11] ,1 can't color 3 but can color it''s neighbours 
// root=6,7,mx=1 
// //[6],[7] we can color 3's neighbour but there won't be a connection like we got 
// //in left subtree

	         


// Example 1:

// Input: root = [1,2,3,4,5,6,7,8,9,10,11], n = 11, x = 3
// Output: true
// Explanation: The second player can choose the node with value 2.

// Example 2:

// Input: root = [1,2,3], n = 3, x = 1
// Output: false

// we maximize our turn & minmize opponents turn by blocking/coloring 
//parent/left/right child nodes which contain maximum number of nodes in 
//it's subtree --> above example if we color 1 (we get 7) , 

// if we can get more than n/2 colored , we win

// 70% faster , 86% less memory
//Time O(N)
//Space O(height) for recursion

class Solution {
private:
    int lc,rc; // how many left & right child can be colored by same color
    int count(TreeNode* root,int x){
        if(!root) return 0;
        int lt=count(root->left,x);
        int rt=count(root->right,x);
        if(root->val==x){ //reached same node which was colored 1st by opponent
            lc=lt;
            rc=rt;
        }
        return lt+rt+1;
    }
public:
    
    bool btreeGameWinningMove(TreeNode* root, int n, int x) {
        //if(!root) return false;
        lc=0;
        rc=0;
        count(root,x);
        int parent=n-(lc+rc+1); //how many node can be colored with color of parent of x
        int mx=max(parent,max(lc,rc));
        return mx>(n/2);
    }
};