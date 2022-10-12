25. Reverse Nodes in k-Group
================================
Given the head of a linked list, reverse the nodes of the list k at a time, 
and return the modified list.k is a positive integer and is less than or equal to 
the length of the linked list. 
If the number of nodes is not a multiple of k then left-out nodes, in the end, 
should remain as it is.

You may not alter the values in the list''s nodes, only nodes themselves may 
be changed.

 

Example 1:

Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Example 2:

Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]

 

Constraints:

    The number of nodes in the list is n.
    1 <= k <= n <= 5000
    0 <= Node.val <= 1000

 

Follow-up: Can you solve the problem in O(1) extra memory space?

//recursive , SC : n (internal) but faster than iterative

Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]


Before reverse ::             head : 1 node : 3
After reverse :: new_head : 2 head : 1 node : 3
Before reverse ::             head : 3 node : 5
After reverse :: new_head : 4 head : 3 node : 5
head->next : 5 node : 5 k : 2
head->next : 4 node : 3 k : 2




ListNode* reverse(ListNode* first,ListNode* last){
        auto prev=last;
        for(;first!=last;){
            auto tmp=first->next;
            first->next=prev;
            prev=first;
            first=tmp;
        }
        return prev;
    }
    ListNode* reverseKGroup(ListNode* head, int k) {
        auto node=head;
        // 1. Generate nodes for the current group;
        for(int i =0;i<k;++i){
            if(!node) return head;
            node=node->next;
        }
        //2. Reverse the nodes for the current group;
        //[(1),2,(3),4,5]-->[2,1,(3),4,(5)]-->[2,1,4,3,5]
        //new_head=2,4|head=1,3|node=3,5
        auto new_head = reverse(head,node); //reverse LL of gr 1,2,k or k-1
        // head->next : 5 node : 5 --> join 3 with 5
        // head->next : 4 node : 3 --> join 1 with 4
        // 3. Reverse the nodes for the next group;
        head->next=reverseKGroup(node,k); //join them
       
        return new_head;
    }

// iterative , slower than recursive

ListNode* reverseKGroup(ListNode* head, int k) {
        if(!head||k==1) return head;
        int num=0;
        ListNode *preheader = new ListNode(-1);
        preheader->next = head;
        ListNode *cur = preheader, *nex, *pre = preheader;
        while(cur = cur->next) 
            num++;
        while(num>=k) {
            cur = pre->next;
            nex = cur->next;
            for(int i=1;i<k;++i) {
                cur->next=nex->next;
                nex->next=pre->next;
                pre->next=nex;
                nex=cur->next;
            }
            pre = cur;
            num-=k;
        }
        return preheader->next;
    }

// Reverse node in K groups

class Node {
public:
    int data;
    Node* next;
};
 
/* Reverses the linked list in groups
of size k and returns the pointer
to the new head node. */
Node* reverse(Node* head, int k)
{
    // base case
    if (!head)
        return NULL;
    Node* current = head;
    Node* next = NULL;
    Node* prev = NULL;
    int count = 0;
 
    /*reverse first k nodes of the linked list */
    while (current != NULL && count < k) {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
        count++;
    }
 
    /* next is now a pointer to (k+1)th node
    Recursively call for the list starting from current.
    And make rest of the list as next of first node */
    if (next != NULL)
        head->next = reverse(next, k);
 
    /* prev is new head of the input list */
    return prev;
}
 
/* UTILITY FUNCTIONS */
/* Function to push a node */
void push(Node** head_ref, int new_data)
{
    /* allocate node */
    Node* new_node = new Node();
 
    /* put in the data */
    new_node->data = new_data;
 
    /* link the old list off the new node */
    new_node->next = (*head_ref);
 
    /* move the head to point to the new node */
    (*head_ref) = new_node;
}

/* Function to print linked list */
void printList(Node* node)
{
    while (node != NULL) {
        cout << node->data << " ";
        node = node->next;
    }
}

int main()
{
    /* Start with the empty list */
    Node* head = NULL;
 
    /* Created Linked list
       is 1->2->3->4->5->6->7->8->9 */
    push(&head, 9);
    push(&head, 8);
    push(&head, 7);
    push(&head, 6);
    push(&head, 5);
    push(&head, 4);
    push(&head, 3);
    push(&head, 2);
    push(&head, 1);
 
    cout << "Given linked list \n";
    printList(head);
    head = reverse(head, 3);
 
    cout << "\nReversed Linked list \n";
    printList(head);
 
    return (0);
}




61. Rotate List
==================
Given the head of a linked list, rotate the list to the right by k places.

 

Example 1:

Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:

Input: head = [0,1,2], k = 4
Output: [2,0,1]

// Brute-force : take last node & put it in the front, TC : k*n (for finding last node)

//optimized
ListNode* rotateRight(ListNode* head, int k) {
        if (!head) return head; //base case
        ListNode* p = head; //p is currently pointing to where head is pointing.
        int n=1; //initial length is 1 since head is not null
        while (p->next){ //to calculate the length of the linked list
            n++;
            p=p->next;
        }
        p->next = head; //p->next is pointing to head, this means now we have a cycle.
        k = k % n; //we take mod when k is greater than n , it rotates accordingly
        for (int i = 0;i<=n-k-1; i++) p=p->next; //we move till n-k-1 , [1,2,3],p=3 : 3 --> 4 --> 5 --> 1 --> 2 --> 3
        head = p->next; //where we want to start after k rotations ,[4,5],head=4 : 4 --> 5 --> 1 --> 2 --> 3 --> 4
        p->next = NULL; //break the cycle. //4 --> 5 --> 1 --> 2 --> 3 --> nullptr 
        return head; //return this new head.
    }

Reverse a Linked List in groups of given size
=================================================
Given a linked list, write a function to reverse every k nodes 
(where k is an input to the function). 

Example: 

    Input: 1->2->3->4->5->6->7->8->NULL, K = 3 
    Output: 3->2->1->6->5->4->8->7->NULL 
    Input: 1->2->3->4->5->6->7->8->NULL, K = 5 
    Output: 5->4->3->2->1->8->7->6->NULL 


Given linked list ,K = 3
1 2 3 4 5 6 7 8 9 
Reversed Linked list 
3 2 1 6 5 4 9 8 7 

// Complexity Analysis: 

//     Time Complexity: O(n). 
//     Traversal of list is done only once and it has ‘n’ elements.
//     Auxiliary Space: O(n/k). 
//     For each Linked List of size n, n/k or (n/k)+1 calls will be made 
//     during the recursion.



/* Reverses the linked list in groups
of size k and returns the pointer
to the new head node. */
Node* reverse(Node* head, int k)
{
    // base case
    if (!head)
        return NULL;
    Node* current = head;
    Node* next = NULL;
    Node* prev = NULL;
    int count = 0;
 
    /*reverse first k nodes of the linked list */
    while (current != NULL && count < k) {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
        count++;
    }
 
    /* next is now a pointer to (k+1)th node
    Recursively call for the list starting from current.
    And make rest of the list as next of first node */
    if (next != NULL)
        head->next = reverse(next, k);
 
    /* prev is new head of the input list */
    return prev;
}

// iterative

// Time Complexity: O(N) : While loop takes O(N/K) time and inner for loop 
// takes O(K) time. So N/K * K = N. Therefore TC O(N)

// Space Complexity: O(1) : No extra space is used.

/* Reverses the linked list in groups
of size k and returns the pointer
to the new head node. */
Node* reverse(Node* head, int k)
{
    // If head is NULL or K is 1 then return head
    if (!head || k == 1)
        return head;
 
    Node* dummy = new Node(); // creating dummy node
    dummy->data = -1;
    dummy->next = head;
 
    // Initializing three points prev, curr, next
    Node *prev = dummy, *curr = dummy, *next = dummy;
 
    // Calculating the length of linked list
    int count = 0;
    while (curr) {
        curr = curr->next;
        count++;
    }
 
    // Iterating till next is not NULL
    while (next) {
        curr = prev->next; // Curr position after every
                           // reverse group
        next = curr->next; // Next will always next to curr
        int toLoop = count > k
                  ? k
                  : count-1; // toLoop will set to count - 1
                             // in case of remaining element
        for (int i = 1; i < toLoop; i++) {
            // 4 steps as discussed above
            curr->next = next->next;
            next->next = prev->next;
            prev->next = next;
            next = curr->next;
        }
        // Setting prev to curr
        prev = curr;
        // Update count
        count -= k;
    }
    // dummy -> next will be our new head for output linked
    // list
    return dummy->next;
}

//using stack

struct Node* Reverse(struct Node* head, int k)
{
    // Create a stack of Node*
    stack<Node*> mystack;
    struct Node* current = head;
    struct Node* prev = NULL;
 
    while (current != NULL) {
 
        // Terminate the loop whichever comes first
        // either current == NULL or count >= k
        int count = 0;
        while (current != NULL && count < k) {
            mystack.push(current);
            current = current->next;
            count++;
        }
 
        // Now pop the elements of stack one by one
        while (mystack.size() > 0) {
 
            // If final list has not been started yet.
            if (prev == NULL) {
                prev = mystack.top();
                head = prev;
                mystack.pop();
            } else {
                prev->next = mystack.top();
                prev = prev->next;
                mystack.pop();
            }
        }
    }
 
    // Next of last element will point to NULL.
    prev->next = NULL;
 
    return head;
}