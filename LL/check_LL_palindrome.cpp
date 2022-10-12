234. Palindrome Linked List
==============================
Given the head of a singly linked list, return true if it is a palindrome.

 

Example 1:

Input: head = [1,2,2,1]
Output: true

Example 2:

Input: head = [1,2]
Output: false


// basic approach traverse list & insert value in a vector + check v[i]==v[n-1-i] 
// for i:0 to n/2-1 , this one is fastest among three

// slowest

    bool isPalindrome(ListNode* head) {
        if (!head or !head->next) return true;
        ListNode* fast = head;
        ListNode* newHead = nullptr;
        while (fast) {
            if (!fast->next) {
                head = head->next;
                break;
            }
            else {
                fast = fast->next->next;
            }
            
            ListNode* nxt = head->next;
            head->next = newHead;
            newHead = head;
            head = nxt;
        }
        
        while (newHead ) {
            if (newHead->val != head->val) return false;
            newHead = newHead->next;
            head = head->next;
        }
        
        return true;
    }

// slower 

class Solution {
private:
    ListNode* ref;  
    bool check(ListNode* node){
        if(node == nullptr) return true;
        bool ans = check(node->next);
        bool isEqual = (ref->val == node->val)? true : false; 
        ref = ref->next;
        return ans && isEqual;
    }
public:
    
    bool isPalindrome(ListNode* head) {
        ref = head;        
        return check(head);
    }
};