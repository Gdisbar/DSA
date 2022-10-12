206. Reverse Linked List
==========================
Given the head of a singly linked list, reverse the list, and return the 
reversed list.

 

Example 1:

Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:

Input: head = [1,2]
Output: [2,1]

Example 3:

Input: head = []
Output: []

//iterative
ListNode* reverseList(ListNode* head) {
        ListNode *prev = NULL, *cur=head, *tmp;
        for(;cur;){
            tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        return prev;
   }


//recursive ,TC: n, SC : n (internal stack)
ListNode* reverseList(ListNode* head) {
       if(!head || !(head->next))  return head;
        auto res = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return res;
    }

// Reverse a DLL



Node* reverseDLL(Node * head)
{
struct Node* next;
struct Node* current = head;
struct Node* prev = NULL;

while(current!=NULL){
    next = current->next;
    current->next = prev;
    current->prev = next;
    prev = current;
    current = next;
}
return prev;

}

92. Reverse Linked List II
==============================
Given the head of a singly linked list and two integers left and right where 
left <= right, reverse the nodes of the list from position left to position right, 
and return the reversed list.

 

Example 1:

Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Example 2:

Input: head = [5], left = 1, right = 1
Output: [5]


ListNode* reverseBetween(ListNode* head, int left, int right) {
        if(!head) return head;
        ListNode* dummy=new ListNode(0); // create a dummy node to mark the head of this list
        dummy->next=head;
        
        ListNode* left_prev=dummy;// make a pointer pre as a marker for the node before reversing
        for(int l=0;l<left-1;++l){
            left_prev=left_prev->next;
        }
        ListNode* prev=left_prev->next; // a pointer to the beginning of a sub-list that will be reversed
        ListNode* cur=left_prev->next->next; // a pointer to a node that will be reversed
        // 1 - 2 -3 - 4 - 5 ; left=2; right =4 ---> left_prev = 1, prev = 2, cur = 3
       // dummy-> 1 -> 2 -> 3 -> 4 -> 5
        for(int i=0;i<right-left;++i){
            auto tmp=cur->next;
            cur->next=prev;
            prev=cur;
            cur=tmp;
        }
        
        left_prev->next->next=cur; //cur=5,prev=4,left_prev=1
        left_prev->next=prev;
        // instead of doing above if you do this it's faster
        // first reversing : dummy->1 - 3 - 2 - 4 - 5; pre = 1, start = 2, then = 4
        // second reversing: dummy->1 - 4 - 3 - 2 - 5; pre = 1, start = 2, then = 5 (finish)
        // for(int i=0;i<right-left;++i){
        //     prev->next=cur->next;
        //     cur->next=left_prev->next;
        //     left_prev->next=cur;
        //     cur=prev->next;
        // }
       
        return dummy->next;
    }