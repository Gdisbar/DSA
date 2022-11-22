Boundary Traversal of binary tree
======================================
// Given a binary tree, print boundary nodes of the binary tree Anti-Clockwise 
// starting from the root. The boundary includes left boundary, leaves, and right 
// boundary in order without duplicate nodes. (The values of the nodes may still be 
// duplicates.)The left boundary is defined as the path from the root to the left-most 
// node. The right boundary is defined as the path from the root to the right-most node. 
// If the root doesn’t have left subtree or right subtree, then the root itself is left 
// boundary or right boundary. 
// Note this definition only applies to the input binary 
// tree, and not apply to any subtrees.The left-most node is defined as a leaf node 
// you could reach when you always firstly travel to the left subtree if it exists. 
// If not, travel to the right subtree. Repeat until you reach a leaf node.
// The right-most node is also defined in the same way with left and right exchanged. 


//                          20
//                        /    \
//                       8     22
//                      / \      \
//                     4   12    25
//                        /  \
//                       10  14

// Boundary Traversal - 20 8 4 10 14 25 22 

// A simple function to print leaf nodes of a binary tree
void printLeaves(Node* root)
{
    if (root == nullptr)
        return;
 
    printLeaves(root->left);
 
    // Print it if it is a leaf node
    if (!(root->left) && !(root->right))
        cout << root->data << " ";
 
    printLeaves(root->right);
}
 
// A function to print all left boundary nodes, except a
// leaf node. Print the nodes in TOP DOWN manner
void printBoundaryLeft(Node* root)
{
    if (root == nullptr)
        return;
 
    if (root->left) {
 
        // to ensure top down order, print the node
        // before calling itself for left subtree
        cout << root->data << " ";
        printBoundaryLeft(root->left);
    }
    else if (root->right) {
        cout << root->data << " ";
        printBoundaryLeft(root->right);
    }
    // do nothing if it is a leaf node, this way we avoid
    // duplicates in output
}
 
// A function to print all right boundary nodes, except a
// leaf node Print the nodes in BOTTOM UP manner
void printBoundaryRight(Node* root)
{
    if (root == nullptr)
        return;
 
    if (root->right) {
        // to ensure bottom up order, first call for right
        // subtree, then print this node
        printBoundaryRight(root->right);
        cout << root->data << " ";
    }
    else if (root->left) {
        printBoundaryRight(root->left);
        cout << root->data << " ";
    }
    // do nothing if it is a leaf node, this way we avoid
    // duplicates in output
}
 
// A function to do boundary traversal of a given binary
// tree
void printBoundary(Node* root)
{
    if (root == nullptr)
        return;
 
    cout << root->data << " ";
 
    // Print the left boundary in top-down manner.
    printBoundaryLeft(root->left);
 
    // Print all leaf nodes
    printLeaves(root->left);
    printLeaves(root->right);
 
    // Print the right boundary in bottom-up manner
    printBoundaryRight(root->right);
}