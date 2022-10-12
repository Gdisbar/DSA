160. Intersection of Two Linked Lists
==========================================
Given the heads of two singly linked-lists headA and headB, return the node 
at which the two lists intersect. If the two linked lists have no intersection 
at all, return null.

// basic approach , 34% faster , 72% less space , no better tricky solution found

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA||!headB) return nullptr;
        int lenA=0,lenB=0; // count length both list
        auto tmpA=headA,tmpB=headB;
        while(tmpA){
            lenA++;
            tmpA=tmpA->next;
        }
        while(tmpB){
            lenB++;
            tmpB=tmpB->next;
        }
        if(lenA>lenB){
            while(lenA!=lenB){  //1st cover the extra length
                lenA--;
                headA=headA->next;
            }
            while(headA&&headB){
                if(headA==headB) return headA;
                headA=headA->next;
                headB=headB->next;
            }
        }
        else if(lenA<lenB){
            while(lenA!=lenB){
                lenB--;
                headB=headB->next;
            }
            while(headA&&headB){
                if(headA==headB) return headB;
                headA=headA->next;
                headB=headB->next;
            }
        }
        else{
            while(headA&&headB){
                if(headA==headB) return headA;
                headA=headA->next;
                headB=headB->next;
            }
        }
        return nullptr;
    }

//slower 
// unordered_set<ListNode*> mp;
//         while(headA) {
//             mp.insert(headA);
//             headA = headA->next;
//         }
//         while(headB) {
//             if (mp.find(headB)!=mp.end()) {
//                 return headB;
//             }
//             headB=headB->next;
//         }