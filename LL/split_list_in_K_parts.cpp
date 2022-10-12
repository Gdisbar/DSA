725. Split Linked List in Parts
=================================
Given the head of a singly linked list and an integer k, split the linked list 
into k consecutive linked list parts.

The length of each part should be as equal as possible: no two parts should 
have a size differing by more than one. This may lead to some parts being null.

The parts should be in the order of occurrence in the input list, and parts 
occurring earlier should always have a size greater than or equal to parts 
occurring later.

Return an array of the k parts.

 

Example 1:

Input: head = [1,2,3], k = 5
Output: [[1],[2],[3],[],[]]
Explanation:
The first element output[0] has output[0].val = 1, output[0].next = null.
The last element output[4] is null, but its string representation as a ListNode 
is [].

Example 2:

Input: head = [1,2,3,4,5,6,7,8,9,10], k = 3
Output: [[1,2,3,4],[5,6,7],[8,9,10]]
Explanation:
The input has been split into consecutive parts with size difference at 
most 1, and earlier parts are a larger size than the later parts.


vector<ListNode*> splitListToParts(ListNode* head, int k) {
        vector<ListNode*> parts(k,nullptr);
        if(!head) return parts;
        auto p=head;
        int n=1;
        for(;p->next;p=p->next)n++;
        int d=n/k,r=n%k;
        ListNode *node=head,*prev=nullptr;
        for(int i=0;i<k&&node;++i,--r){
            parts[i]=node;
            //we need to decrement r by 1 for each part & add 1 new node for that
            for (int j=0;j<d+(r>0);++j){ //d+(r>0) , if r>0 then it''s d+1 otherwise d
                prev=node;               //step 1 of two pointer
                node=node->next;         //step 2 of two pointer
            }
            prev->next=nullptr;         //step 3 of two pointer
        }
        return parts;
    }
