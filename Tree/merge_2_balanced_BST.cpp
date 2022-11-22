Merge Two Balanced Binary Search Trees
========================================
// You are given two balanced binary search trees e.g., AVL or Red-Black Tree. 
// Write a function that merges the two given balanced BSTs into a balanced binary 
// search tree. Let there be m elements in the first tree and n elements in the other 
// tree. Your merge function should take O(m+n) time.

// Method 1 (Insert elements of the first tree to second):

// Take all elements of first BST one by one, and insert them into the second BST. 
// Inserting an element to a self balancing BST takes Logn time (See this) where n 
// is size of the BST. So time complexity of this method is 
// Log(n) + Log(n+1) … Log(m+n-1). The value of this expression will be between mLogn 
// and mLog(m+n-1). As an optimization, we can pick the smaller tree as first tree.

// Method 2 (Merge Inorder Traversals):

// Do inorder traversal of first tree and store the traversal in one temp array arr1[]. 
// This step takes O(m) time. 
// Do inorder traversal of second tree and store the traversal in another temp array 
// arr2[]. This step takes O(n) time. 
// The arrays created in step 1 and 2 are sorted arrays. Merge the two sorted arrays 
// into one array of size m + n. This step takes O(m+n) time. 
// Construct a balanced tree from the merged array using the technique discussed in 
// this post. This step takes O(m+n) time.

// Time complexity of this method is O(m+n) which is better than method 1. 
// This method takes O(m+n) time even if the input BSTs are not balanced. 

 /* 
        100                80
        / \               /  \               
       50 300    +       40   120
       / \
      20 70
    
						  100                
                          /  \               
                         50   120
      		    		/ \     \
                       20  70   300
                        \   \
                        40  80
     

    */


// // A helper function that stores inorder
// // traversal of a tree rooted with node
// void storeInorder(node* node, int inorder[], int *index_ptr)
// {
//     if (node == NULL)
//         return;
 
//     /* first recur on left child */
//     storeInorder(node->left, inorder, index_ptr);
 
//     inorder[*index_ptr] = node->data;
//     (*index_ptr)++; // increase index for next entry
 
//     /* now recur on right child */
//     storeInorder(node->right, inorder, index_ptr);
// }


/* This function merges two balanced
BSTs with roots as root1 and root2.
m and n are the sizes of the trees respectively */
node* mergeTrees(node *root1, node *root2, int m, int n)
{
    // Store inorder traversal of
    // first tree in an array arr1[]
    int *arr1 = new int[m];
    int i = 0;
    storeInorder(root1, arr1, &i);
 
    // Store inorder traversal of second
    // tree in another array arr2[]
    int *arr2 = new int[n];
    int j = 0;
    storeInorder(root2, arr2, &j);
 
    // Merge the two sorted array into one using merge fn of merge sort
    int *mergedArr = merge(arr1, arr2, m, n);
 
    // Construct a tree from the merged
    // array and return root of the tree
    return sortedArrayToBST (mergedArr, 0, m + n - 1);
}
 
 
/* A function that constructs Balanced
// Binary Search Tree from a sorted array
See https://www.geeksforgeeks.org/sorted-array-to-balanced-bst/ */
node* sortedArrayToBST(int arr[], int start, int end)
{
    /* Base Case */
    if (start > end)
    return NULL;
 
    /* Get the middle element and make it root */
    int mid = (start + end)/2;
    /* Helper function that allocates a new node with the given data and
     NULL left and right pointers. */
    node *root = newNode(arr[mid]);
 
    /* Recursively construct the left subtree and make it
    left child of root */
    root->left = sortedArrayToBST(arr, start, mid-1);
 
    /* Recursively construct the right subtree and make it
    right child of root */
    root->right = sortedArrayToBST(arr, mid+1, end);
 
    return root;
}


# A binary tree node has data, pointer to left child 
# and a pointer to right child
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
 
# A utility function to merge two sorted arrays into one
# Time Complexity of below function: O(m + n)
# Space Complexity of below function: O(m + n)
def merge_sorted_arr(arr1, arr2):
    arr = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            arr.append(arr1[i])
            i += 1
        else:
            arr.append(arr2[j])
            j += 1
    while i < len(arr1):
        arr.append(arr1[i])
        i += 1
    while i < len(arr2):
        arr.append(arr2[j])
        j += 1
    return arr
 
# A helper function that stores inorder
# traversal of a tree in arr
def inorder(root, arr = []):
    if root:
        inorder(root.left, arr)
        arr.append(root.val)
        inorder(root.right, arr)
 
# A utility function to insert the values
# in the individual Tree
def insert(root, val):
    if not root:
        return Node(val)
    if root.val == val:
        return root
    elif root.val > val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root
 
# Converts the merged array to a balanced BST
# Explanation of the below code:
# https://www.geeksforgeeks.org/sorted-array-to-balanced-bst/
def arr_to_bst(arr):
    if not arr:
        return None
    mid = len(arr) // 2
    root = Node(arr[mid])
    root.left = arr_to_bst(arr[:mid])
    root.right = arr_to_bst(arr[mid + 1:])
    return root
 
if __name__=='__main__':
    root1 = root2 = None
     
    # Inserting values in first tree
    root1 = insert(root1, 100)
    root1 = insert(root1, 50)
    root1 = insert(root1, 300)
    root1 = insert(root1, 20)
    root1 = insert(root1, 70)
     
    # Inserting values in second tree
    root2 = insert(root2, 80)
    root2 = insert(root2, 40)
    root2 = insert(root2, 120)
    arr1 = []
    inorder(root1, arr1)
    arr2 = []
    inorder(root2, arr2)
    arr = merge_sorted_arr(arr1, arr2)
    root = arr_to_bst(arr)
    res = []
    inorder(root, res)
    print('Following is Inorder traversal of the merged tree')
    for i in res:
      print(i, end = ' ')