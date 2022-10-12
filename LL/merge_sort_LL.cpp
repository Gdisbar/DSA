148. Sort List
===================
Given the head of a linked list, return the list after sorting it in ascending order.

 

Example 1:

Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:

Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Example 3:

Input: head = []
Output: []


class Solution {
    //MergeSort Function O(n*logn) , SC : each function call to sortList takes O(1) memory on the stack, 
    //and there are O(log n) calls to sortList, so total memory consumed on stack is O(log n).
    ListNode *mergeSort(ListNode *l1, ListNode *l2)
    {
        ListNode dummy;
        ListNode *curr = &dummy;
        
        while (l1 && l2)
        {
            if (l1->val <= l2->val)
            {
                curr->next = l1;
                l1 = l1->next;
            }
            else {
                curr->next = l2;
                l2 = l2->next;
            }
            
            curr = curr->next;
        }
        
        if (l1)
            curr->next = l1;
        if (l2)
            curr->next = l2;
        
        return dummy.next;
    }
public:
    ListNode* sortList(ListNode* head) {
        if (head == NULL || head->next == NULL)
            return head;
        
        ListNode *slow = head;
        ListNode *fast = head;
        ListNode *prev = NULL;
        // 2 pointer appraoach / turtle-hare Algorithm (Finding the middle element)
        while (fast && fast->next)
        {
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        
        prev->next = NULL;   //end of first left half
        
        ListNode *l1 = sortList(head); //left half recursive call
        ListNode *l2 = sortList(slow); //right half recursive call
        
        return mergeSort(l1, l2);   //mergelist Function call
    }
};


// Quicksort // very slow will give TLE for range [0, 5 * 104] 

class Solution {
private:
    void SwapValue (ListNode* a ,ListNode* b){
        int temp = a->val;
        a->val = b->val;
        b->val = temp;
        
    }
    
    
    ListNode* Partition(ListNode* start ,ListNode* end){
        int pivotValue = start->val;
        ListNode* p = start;
        ListNode* q = start -> next;
        while(q != end){
            if (q -> val < pivotValue){
                p = p -> next;
                SwapValue(p,q);
            }
            q = q -> next;
        }
        SwapValue(p,start);
        return p;
    }
        
        
    void QuickSort(ListNode* start ,ListNode* end){
        if (start != end){
            ListNode* mid =Partition(start ,end);
            QuickSort(start,mid);
            QuickSort(mid->next,end);
        }
        
    }

public:
     ListNode* sortList(ListNode* head) {
        QuickSort(head,NULL);
        return head;
    }
};