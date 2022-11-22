Construct a tree from Inorder and Level order traversals
=========================================================
// Input: Two arrays that represent Inorder and level order traversals of a 
// Binary Tree
// in[]    = {4, 8, 10, 12, 14, 20, 22};
// level[] = {20, 8, 22, 4, 12, 10, 14};

// Output: Construct the tree represented by the two arrays.
//         For the above two arrays, the constructed tree is shown in 
//         the diagram on right side


//  									20
//  								  /    \
//  								 8     22
//  							   /  \
//  							  4   12 
//  							     /  \
//  							    10  14


//   			 20
//            /    \
//           /      \ 
//  {4,8,10,12,14}  {22}

//    			 20                
//            /    \
//           /      \ 
//          8      {22}              
//        /  \
//       /    \
//     {4}   {10,12,14}


/* A binary tree node */
struct Node {
    int key;
    struct Node *left, *right;
};
 
/* Function to find index of value in arr[start...end] */
int search(int arr[], int strt, int end, int value)
{
    for (int i = strt; i <= end; i++)
        if (arr[i] == value)
            return i;
    return -1;
}
 
// n is size of level[], m is size of in[] and m < n. This
// function extracts keys from level[] which are present in
// in[].  The order of extracted keys must be maintained
int* extrackKeys(int in[], int level[], int m, int n)
{
    int *newlevel = new int[m], j = 0;
    for (int i = 0; i < n; i++)
        if (search(in, 0, m - 1, level[i]) != -1)
            newlevel[j] = level[i], j++;
    return newlevel;
}
 
/* function that allocates a new node with the given key  */
Node* newNode(int key)
{
    Node* node = new Node;
    node->key = key;
    node->left = node->right = NULL;
    return (node);
}
 
/* Recursive function to construct binary tree of size n
   from Inorder traversal in[] and Level Order traversal
   level[]. inStrt and inEnd are start and end indexes of
   array in[] Initial values of inStrt and inEnd should be 0
   and n -1. The function doesn't do any error checking for
   cases where inorder and levelorder do not form a tree */
Node* buildTree(int in[], int level[], int inStrt,int inEnd, int n)
{
 
    // If start index is more than the end index
    if (inStrt > inEnd)
        return NULL;
 
    /* The first node in level order traversal is root */
    Node* root = newNode(level[0]);
 
    /* If this node has no children then return */
    if (inStrt == inEnd)
        return root;
 
    /* Else find the index of this node in Inorder traversal
     */
    int inIndex = search(in, inStrt, inEnd, root->key);
 
    // Extract left subtree keys from level order traversal
    int* llevel = extrackKeys(in, level, inIndex, n);
 
    // Extract right subtree keys from level order traversal
    int* rlevel
        = extrackKeys(in + inIndex + 1, level, n - 1, n);
 
    /* construct left and right subtrees */
    root->left = buildTree(in, llevel, inStrt, inIndex - 1,
                           inIndex - inStrt);
    root->right = buildTree(in, rlevel, inIndex + 1, inEnd,
                            inEnd - inIndex);
 
    // Free memory to avoid memory leak
    delete[] llevel;
    delete[] rlevel;
 
    return root;
}
 
/* utility function to print inorder traversal of binary
 * tree */
void printInorder(Node* node)
{
    if (node == NULL)
        return;
    printInorder(node->left);
    cout << node->key << " ";
    printInorder(node->right);
}    