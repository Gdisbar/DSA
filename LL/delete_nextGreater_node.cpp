1019. Next Greater Node In Linked List
==========================================
You are given the head of a linked list with n nodes.

For each node in the list, find the value of the next greater node. 
That is, for each node, find the value of the first node that is next to it and 
has a strictly larger value than it.

Return an integer array answer where answer[i] is the value of the next greater 
node of the ith node (1-indexed). If the ith node does not have a next greater node, 
set answer[i] = 0.

 

Example 1:

Input: head = [2,1,5]
Output: [5,5,0]

Example 2:

Input: head = [2,7,4,3,5]
Output: [7,0,5,5,0]

Push nodes values to vector<int> res.
vector<int> stack will save the indices of elements that need to find next 
greater element.
In the end, we reset 0 to all elements that have no next greater elements.


  vector<int> nextLargerNodes(ListNode* head) {
        vector<int> res, stack;
        for (ListNode* node = head; node; node = node->next) {
            while (stack.size() && res[stack.back()] < node->val) {
                res[stack.back()] = node->val;
                stack.pop_back();
            }
            stack.push_back(res.size());
            res.push_back(node->val);
        }
        for (int i: stack) res[i] = 0;
        return res;
    }

Delete nodes having greater value on right
=============================================
Given a singly linked list, remove all the nodes which have a greater value 
on their right side.

Example 1:

Input:
LinkedList = 12->15->10->11->5->6->2->3
Output: 15 11 6 3
Explanation: Since, 12, 10, 5 and 2 are
the elements which have greater elements
on the following nodes. So, after deleting
them, the linked list would like be 15,
11, 6, 3.

Example 2:

Input:
LinkedList = 10->20->30->40->50->60
Output: 60




Node* reverse(Node* head) {
        Node* prev = nullptr;
        
        for(Node* cur=head;cur;){
            Node* tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        return prev;
    }
    Node *compute(Node *head)
    {
        // your code goes here
        head=reverse(head);
        int mx=head->data;
        Node* prev=head; //prev=3
        Node* cur=head;  
        head=head->next; //head=2
        while(head){
            if(head->data>=mx){ //doesn't have greater value ,will be included in answer
                mx=head->data;
                prev=head;
                head=head->next;
            }
            else{
                prev->next=head->next; //prev=2,head=6 --> prev->next=5
                head=prev->next; //head=5
            }
            
        }
        head=reverse(cur);
        return head;
    }