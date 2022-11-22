In-place convert a binary tree to a doubly-linked list
=======================================================
// The conversion should be done such that the left and right pointers of binary 
// tree nodes should act as previous and next pointers in a doubly-linked list, 
// and the doubly linked list nodes should follow the same order of nodes as 
// inorder traversal on the tree.


// TC : n, SC : h
    /* Construct the following tree
              1
           /    \
          2      3
         / \    / \
        4   5  6   7
    */
//4 2 5 1 6 3 7
// Function to in-place convert a given binary tree into a doubly linked list
// by doing reverse inorder traversal
void convert(Node* root, Node* &head)
{
    // base case: tree is empty
    if (root == nullptr) {
        return;
    }
 
    // recursively convert the right subtree first
    convert(root->right, head);
 
    // insert the current node at the beginning of a doubly linked list
    root->right = head;
 
    if (head != nullptr) {
        head->left = root;
    }
 
    head = root;
 
    // recursively convert the left subtree
    convert(root->left, head);
}
 
// In-place convert a given binary tree into a doubly linked list
void convert(Node* root)
{
    // head of the doubly linked list
    Node* head = nullptr;
 
    // convert the above binary tree into doubly linked list
    convert(root, head);
 
    // print the list
    printDLL(head);
}



Convert a Binary Tree to a Circular Doubly Link List
=========================================================

// A function that appends rightList at the end
// of leftList.
Node *concatenate(Node *leftList, Node *rightList)
{
    // If either of the list is empty
    // then return the other list
    if (leftList == NULL)
        return rightList;
    if (rightList == NULL)
        return leftList;
 
    // Store the last Node of left List
    Node *leftLast = leftList->left;
 
    // Store the last Node of right List
    Node *rightLast = rightList->left;
 
    // Connect the last node of Left List
    // with the first Node of the right List
    leftLast->right = rightList;
    rightList->left = leftLast;
 
    // Left of first node points to
    // the last node in the list
    leftList->left = rightLast;
 
    // Right of last node refers to the first
    // node of the List
    rightLast->right = leftList;
 
    return leftList;
}
 
// Function converts a tree to a circular Linked List
// and then returns the head of the Linked List
Node *bTreeToCList(Node *root)
{
    if (root == NULL)
        return NULL;
 
    // Recursively convert left and right subtrees
    Node *left = bTreeToCList(root->left);
    Node *right = bTreeToCList(root->right);
 
    // Make a circular linked list of single node
    // (or root). To do so, make the right and
    // left pointers of this node point to itself
    root->left = root->right = root;
 
    // Step 1 (concatenate the left list with the list
    //         with single node, i.e., current node)
    // Step 2 (concatenate the returned list with the
    //         right List)
    return concatenate(concatenate(left, root), right);
}
 
// Display Circular Link List
void displayCList(Node *head)
{
    cout << "Circular Linked List is :\n";
    Node *itr = head;
    do
    {
        cout << itr->data <<" ";
        itr = itr->right;
    } while (head!=itr);
    cout << "\n";
}
