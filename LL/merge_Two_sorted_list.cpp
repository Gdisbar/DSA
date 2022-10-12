21. Merge Two Sorted Lists
============================  
You are given the heads of two sorted linked lists list1 and list2.Merge the two 
lists in a one sorted list. The list should be made by splicing together the nodes 
of the first two lists.

Return the head of the merged linked list.

 

Example 1:

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:

Input: list1 = [], list2 = []
Output: []

Example 3:

Input: list1 = [], list2 = [0]
Output: [0]


//11% faster,81% less memory , TC : m+n

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* res=nullptr;
        if(!list1) return list2;
        else if(!list2) return list1;
        if(list1->val<=list2->val){
            res=list1;
            res->next=mergeTwoLists(list1->next,list2);
        }
        else{
            res=list2;
            res->next=mergeTwoLists(list1,list2->next);
        }
        return res;
    }

// faster than 94%,42% less memory ,iterative approach of above 
// My was ptr update was wrong

If you do Node tmp, then tmp is destroyed after you exit the scope in which it 
was defined.
On the other hand, doing Node *tmp = new Node allocates memory for a Node on the 
heap and gives you a pointer tmp to it. The Node is only destroyed if and when 
you call delete on it.

you would want to pass a pointer by reference if you have a need to modify the 
pointer rather than the object that the pointer is pointing to.

If you do head = tmp, tail = tmp without passing head and tail by reference, 
then head and tail will only retain this value until the scope of this function. 
This change will not reflect outside of addNode().



In order to modify head and tailusing function, they must be passed by 
reference i.e second code snippet. In first snippet, they were passed by value, 
any modification has done inside function addNode() will be vanished after the 
function out of scope.

case 1: pass by value
-------------------------
addNode(head);   // call

function body:
----------------
void addNode(Node *headCopy) {   // Node *headCopy = head,   copy is created
    headCopy = &someThing;    // the copy of head point to something else
                              // not the actual 'head' that changes its pointee
}

case 2: pass by reference
---------------------------
addNode(&head);

function body:
-------------------
void addNode(Node *&headReference) {    // reference to 'head'
    headReference = &someThing;  // the actual 'head' changes its pointee
}





ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(list1 == nullptr) return list2;
        if(list2 == nullptr) return list1;
        ListNode* ptr = list1;
        if(list1->val>list2->val){
            ptr = list2;
            list2 = list2 -> next;
        }
        else{
            list1 = list1 -> next;
        }
     //I was not updating ptr after head insertion +
    //now ptr is head we need a pointer to head but I was increasing head(ptr) itself
        ListNode *curr=ptr; // makes execution faster due to heap memory
        while(list1&&list2){ // till one of the list doesn't reaches NULL
            if(list1->val<list2->val){
                curr->next = list1;
                list1 = list1 -> next;
            }
            else{
                curr->next = list2;
                list2 = list2 -> next;
            }
            curr = curr -> next;
        }
		// adding remaining elements of bigger list. // I was using while here
        if(!list1) curr -> next = list2;
        else curr -> next = list1;
        return ptr;
        
    }
// slightly faster but implementation style is important

/* UTILITY FUNCTIONS */
/* MoveNode() function takes the
node from the front of the source,
and move it to the front of the dest.
It is an error to call this with the
source list empty.
 
Before calling MoveNode():
source == {1, 2, 3}
dest == {1, 2, 3}
 
After calling MoveNode():
source == {2, 3}
dest == {1, 1, 2, 3} */

 class Solution {
private:
    void MoveNode(ListNode** dest,ListNode** src){
        ListNode* tmp=*src; // the front source node 
        assert(tmp!=nullptr); //can't apply on null
        *src=tmp->next; // Advance the source pointer 
        tmp->next=*dest; // Link the old dest off the new node 
        *dest=tmp;     // Move dest to point to the new node 
    }
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* res=nullptr;
        ListNode** last=&res; //points to the last pointer of the result list.
        if(!list1) return list2;
        else if(!list2) return list1;
        while (1) {
        if (!list1) {
            *last=list2;
            break;
        }
        else if(!list2) {
            *last=list1;
            break;
        }
        if (list1->val<=list2->val){
            MoveNode(last,&list1);
        }
        else MoveNode(last,&list2);
        last =&((*last)->next); //tricky: advance to point to the next  
    }
        return res;
    }
};



// /* Link list node */
// class Node
// {
//     public:
//     int data;
//     Node* next;
// };
// void push(Node** head_ref, int new_data)
// {
//     /* allocate node */
//     Node* new_node = new Node();
 
//     /* put in the data */
//     new_node->data = new_data;
 
//     /* link the old list off the new node */
//     new_node->next = (*head_ref);
 
//      move the head to point to the new node 
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
// /* Driver code*/
// int main()
// {
//     /* Start with the empty list */
//     Node* res = NULL;
//     Node* a = NULL;
//     Node* b = NULL;
 
//     /* Let us create two sorted linked lists 
//     to test the functions
//     Created lists, a: 5->10->15, b: 2->3->20 */
//     push(&a, 15);
//     push(&a, 10);
//     push(&a, 5);
 
//     push(&b, 20);
//     push(&b, 3);
//     push(&b, 2);
 
//     /* Remove duplicates from linked list */
//     res = SortedMerge(a, b);
 
//     cout << "Merged Linked List is: \n";
//     printList(res);
 
//     return 0;
// }