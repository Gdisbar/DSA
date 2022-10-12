876. Middle of the Linked List
================================
Given the head of a singly linked list, return the middle node of the 
linked list.
If there are two middle nodes, return the second middle node.

 

Example 1:

Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

Example 2:

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, 
we return the second one.


ListNode* middleNode(ListNode* head) {
        auto slow=head,fast=head;
        for(;fast!=nullptr&&fast->next!=nullptr;){
            slow=slow->next;
            fast=fast->next->next;
        }
        return slow;
    }

141. Linked List Cycle
======================= 
Given head, the head of a linked list, determine if the linked list has a 
cycle in it.

There is a cycle in a linked list if there is some node in the list that can 
be reached again by continuously following the next pointer. Internally, 
pos is used to denote the index of the node that tail''s next pointer is 
connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to 
the 1st node (0-indexed).

Example 2:

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to 
the 0th node.

Example 3:

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.



bool hasCycle(ListNode *head) {
        if(!head) return false;
        auto slow=head,fast=head;
        for(;fast&&fast->next;){ //sometime we need to add slow!=nullptr
            slow=slow->next;
            fast=fast->next->next;
            //we can't add this line at 1st in that case initalization condition is met & always true is returned
            if(slow==fast) return true; 
        }
        return false;
    }


142. Linked List Cycle II
===============================
Given the head of a linked list, return the node where the cycle begins. 
If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can 
be reached again by continuously following the next pointer. Internally, 
pos is used to denote the index of the node that tail''s next pointer is 
connected to (0-indexed). It is -1 if there is no cycle. Note that pos is 
not passed as a parameter.

Do not modify the linked list.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the 
second node.

Example 2:

Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the 
first node.

Example 3:

Input: head = [1], pos = -1
Output: no cycle
Explanation: There is no cycle in the linked list.


L1 = head and entry , L2 = entry and the meeting , C = length of the cycle , 
n = travel times of the fast around the cycle When the first encounter 
of the slow  and the fast  

when slow & fast encounter
    slow = L1 + L2
    fast =  L1 + L2 + n * C
    2 * (L1+L2) = L1 + L2 + n * C 
    => L1 + L2 = n * C 
    => L1 = (n - 1) C + (C - L2)*

It can be concluded that the distance between the head location and 
entry location is equal to the distance between the meeting location and the 
entry location along the direction of forward movement.


So, when the slow pointer and the fast pointer encounter in the cycle, 
we can define a pointer "entry" that point to the head, this "entry" 
pointer moves one step each time so as the slow pointer. When this "entry" 
pointer and the slow pointer both point to the same location, this location 
is the node where the cycle begins.



ListNode *detectCycle(ListNode *head) {
        if(!head) return nullptr;
        auto slow = head, fast = head;
        for(;fast && fast->next;) {
            slow = slow->next, fast = fast->next->next;
            if (slow == fast) { // -4,last node,i.e slow travel n & fast 2*n
                slow = head;
                for(;slow != fast;) { //slow = head->cycle point, fast = end->cycle point
                    slow = slow->next, fast = fast->next;
                }
                return slow;
            }
        }
        return nullptr;
    }