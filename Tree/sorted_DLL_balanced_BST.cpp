In-place conversion of Sorted DLL to Balanced BST
=====================================================
// Given a Doubly Linked List which has data members sorted in ascending order. 
// Construct a Balanced Binary Search Tree which has same data members as the given 
// Doubly Linked List. The tree must be constructed in-place (No new node should be 
// allocated for tree conversion) 

// Examples: 

// Input:  Doubly Linked List 1  2  3
// Output: A Balanced BST 
//      2   
//    /  \  
//   1    3 


// Input: Doubly Linked List 1  2 3  4 5  6  7
// Output: A Balanced BST
//         4
//       /   \
//      2     6
//    /  \   / \
//   1   3  5   7  

// Input: Doubly Linked List 1  2  3  4
// Output: A Balanced BST
//       3   
//     /  \  
//    2    4 
//  / 
// 1

// Input:  Doubly Linked List 1  2  3  4  5  6
// Output: A Balanced BST
//       4   
//     /   \  
//    2     6 
//  /  \   / 
// 1   3  5   

// Method 1 (Simple) 
// Following is a simple algorithm where we first find the middle node of list and 
// make it root of the tree to be constructed.  

// 1) Get the Middle of the linked list and make it root.
// 2) Recursively do same for left half and right half.
//        a) Get the middle of left half and make it left child of the root
//           created in step 1.
//        b) Get the middle of right half and make it right child of the
//           root created in step 1.

// Method 2 (Tricky) 
// The method 1 constructs the tree from root to leaves. In this method, we construct 
// from leaves to root. The idea is to insert nodes in BST in the same order as they 
// appear in Doubly Linked List, so that the tree can be constructed in O(n) time 
// complexity. We first count the number of nodes in the given Linked List. 
// Let the count be n. After counting nodes, we take left n/2 nodes and recursively 
// construct the left subtree. After left subtree is constructed, we assign middle 
// node to root and link the left subtree with root. Finally, we recursively construct 
// the right subtree and link it with root. 
// While constructing the BST, we also keep moving the list head pointer to next so 
// that we have the appropriate pointer in each recursive call. 

// /* A Doubly Linked List node that
// will also be used as a tree node */
// class Node 
// { 
//     public:
//     int data; 
  
//     // For tree, next pointer can be
//     // used as right subtree pointer 
//     Node* next; 
  
//     // For tree, prev pointer can be
//     // used as left subtree pointer 
//     Node* prev; 
// }; 
  
// /* UTILITY FUNCTIONS */
// /* A utility function that returns 
// count of nodes in a given Linked List */
// int countNodes(Node *head) 
// { 
//     int count = 0; 
//     Node *temp = head; 
//     while(temp) 
//     { 
//         temp = temp->next; 
//         count++; 
//     } 
//     return count; 
// } 
  
/* The main function that constructs 
balanced BST and returns root of it. 
head_ref --> Pointer to pointer to
head node of Doubly linked list 
n --> No. of nodes in the Doubly Linked List */
Node* sortedListToBSTRecur(Node **head_ref, int n) 
{ 
    /* Base Case */
    if (n <= 0) 
        return NULL; 
  
    /* Recursively construct the left subtree */
    Node *left = sortedListToBSTRecur(head_ref, n/2); 
  
    /* head_ref now refers to middle node,
    make middle node as root of BST*/
    Node *root = *head_ref; 
  
    // Set pointer to left subtree 
    root->prev = left; 
  
    /* Change head pointer of Linked List
    for parent recursive calls */
    *head_ref = (*head_ref)->next; 
  
    /* Recursively construct the right 
    subtree and link it with root 
    The number of nodes in right subtree
    is total nodes - nodes in 
    left subtree - 1 (for root) */
    root->next = sortedListToBSTRecur(head_ref, n-n/2-1); 
  
    return root; 
}
  
/* This function counts the number of 
nodes in Linked List and then calls 
sortedListToBSTRecur() to construct BST */
Node* sortedListToBST(Node *head) 
{ 
    /*Count the number of nodes in Linked List */
    int n = countNodes(head); 
  
    /* Construct BST */
    return sortedListToBSTRecur(&head, n); 
} 
  
 
  

  
// /* Function to insert a node at 
// the beginning of the Doubly Linked List */
// void push(Node** head_ref, int new_data) 
// { 
//     /* allocate node */
//     Node* new_node = new Node();
  
//     /* put in the data */
//     new_node->data = new_data; 
  
//     /* since we are adding at the beginning, 
//     prev is always NULL */
//     new_node->prev = NULL; 
  
//     /* link the old list off the new node */ new_node -> new_node (previous,head)
//     new_node->next = (*head_ref); 
  
//     /* change prev of head node to new node */ new_node <- new_node (previous,head)
//     if((*head_ref) != NULL) 
//     (*head_ref)->prev = new_node ; 
  
//     /* move the head to point to the new node */
//     (*head_ref) = new_node; 
// } 
  
// /* Function to print nodes in a given linked list */
// void printList(Node *node) 
// { 
//     while (node!=NULL) 
//     { 
//         cout<<node->data<<" "; 
//         node = node->next; 
//     } 
// } 
  
// /* A utility function to print
// preorder traversal of BST */
// void preOrder(Node* node) 
// { 
//     if (node == NULL) 
//         return; 
//     cout<<node->data<<" "; 
//     preOrder(node->prev); 
//     preOrder(node->next); 
// } 
  
// /* Driver code*/
// int main() 
// { 
//     /* Start with the empty list */
//     Node* head = NULL; 
  
//     /* Let us create a sorted linked list to test the functions 
//     Created linked list will be 7->6->5->4->3->2->1 */
//     push(&head, 7); 
//     push(&head, 6); 
//     push(&head, 5); 
//     push(&head, 4); 
//     push(&head, 3); 
//     push(&head, 2); 
//     push(&head, 1); 
  
//     cout<<"Given Linked List\n"; 
//     printList(head); 
  
//     /* Convert List to BST */
//     Node *root = sortedListToBST(head); 
//     cout<<"\nPreOrder Traversal of constructed BST \n "; 
//     preOrder(root); 
  
//     return 0; 
// } 
  

Output: 

Given Linked List 
1 2 3 4 5 6 7 
Pre-Order Traversal of constructed BST 
4 2 1 3 6 5 7 


.