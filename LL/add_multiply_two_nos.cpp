2. Add Two Numbers
====================
You are given two non-empty linked lists representing two non-negative integers. 
The digits are stored in reverse order, and each of their nodes contains a 
single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the 
number 0 itself.

 

Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]


ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    int c = 0;
    ListNode newHead(0);
    ListNode *t = &newHead;
    while(c || l1 || l2) {
        c += (l1? l1->val : 0) + (l2? l2->val : 0);
        t->next = new ListNode(c%10);
        t = t->next;
        c /= 10;
        if(l1) l1 = l1->next;
        if(l2) l2 = l2->next;
    }
    return newHead.next;
}

//Recursive

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    if(!l1 && !l2) return NULL;
    int c = (l1? l1->val:0) + (l2? l2->val:0);
    ListNode *newHead = new ListNode(c%10), *next = l1? l1->next:NULL;
    c /= 10;
    if(next) next->val += c;
    else if(c) next = new ListNode(c);
    newHead->next = addTwoNumbers(l2? l2->next:NULL, next);
    return newHead;
}



445. Add Two Numbers II
==========================
You are given two non-empty linked lists representing two non-negative integers. 
The most significant digit comes first and each of their nodes contains a 
single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the 
number 0 itself.

 

Example 1:

Input: l1 = [7,2,4,3], l2 = [5,6,4]
Output: [7,8,0,7]

Example 2:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [8,0,7]

Example 3:

Input: l1 = [0], l2 = [0]
Output: [0]


class Solution {
public:
int getLength(ListNode* l)
{
    int len = 0;
    while(l)
        len++, l = l->next;
    return len;
}

//Adding nodes of (LL1 & LL2 + carry) from RHS to LHS
int recur(ListNode* h1, ListNode* h2)
{
    if(!h2) return 0;
    int carry = recur(h1->next, h2->next);
    
    h1->val += h2->val + carry;
    carry = h1->val/10;
    h1->val %= 10;
    
    return carry;
}

//Adding LL1 + carry from RHS to LHS
int recur2(ListNode* h1, ListNode* end)
{
    if(h1 == end)   return 1;       //bcoz carry was present
    int carry = recur2(h1->next, end);
    
    h1->val += carry;
    carry = h1->val/10;
    h1->val %= 10;
    
    return carry;
}

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) 
{
    int len1 = getLength(l1), len2 = getLength(l2);
    
    if(len2 > len1)
        swap(len1, len2), swap(l1, l2);
    
    //************************Same as finding nth node from RHS of LL
    ListNode* next1 = l1, *prev = l1, *next2 = l2;
    while(next2)
        next1 = next1->next, next2 = next2->next;
    
    while(next1)
        prev = prev->next, next1 = next1->next;
    //*************************
    
    ListNode* head = l1;
    if(recur(prev, l2))             //if carry is present
        if(recur2(l1, prev))        //if again carry is present, make new node
            head = new ListNode(1), head->next = l1;
    
    return head;
}

};

//recursion

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int n1 = length(l1), n2 = length(l2);
        int carry = 0;
        ListNode* h = new ListNode(1);
        h->next = n1 > n2? add_aux(l1, l2, n1-n2, carry):add_aux(l2, l1, n2-n1, carry);
        return carry == 1? h:h->next;
    }
private:
    int length(ListNode* l) {
        int len = 0;
        while (l != nullptr) {
            len++;
            l = l->next;
        }
        return len;
    }
    ListNode* add_aux(ListNode* l1, ListNode* l2, int k, int& carry) {
        if (l2 == nullptr) return nullptr;
        ListNode* p = new ListNode(l1->val);
        if (k > 0) {
            p->next = add_aux(l1->next, l2, k-1, carry);
        }
        else {
            p->val += l2->val;
            p->next = add_aux(l1->next, l2->next, k, carry);
        }
        p->val += carry;
        carry = p->val/10;
        p->val %= 10;
        return p;
    }
};

Add two numbers represented by linked lists | Set-1
=====================================================
Given two numbers represented by two lists, write a function that returns the 
sum list. The sum list is a list representation of the addition of two input 
numbers.

Example:

    Input: 
    List1: 5->6->3 // represents number 563 
    List2: 8->4->2 // represents number 842 
    Output: 
    Resultant list: 1->4->0->5 // represents number 1405 
    Explanation: 563 + 842 = 1405

    Input: 
    List1: 7->5->9->4->6 // represents number 75946
    List2: 8->4 // represents number 84
    Output: 
    Resultant list: 7->6->0->3->0// represents number 76030
    Explanation: 75946+84=76030

class Node {
public:
    int data;
    Node* next;
};
 
/* Function to create a
new node with given data */
Node* newNode(int data)
{
    Node* new_node = new Node();
    new_node->data = data;
    new_node->next = NULL;
    return new_node;
}
 
/* Function to insert a node at the
beginning of the Singly Linked List */
void push(Node** head_ref, int new_data)
{
    /* allocate node */
    Node* new_node = newNode(new_data);
    /* link the old list off the new node */
    new_node->next = (*head_ref);
    /* move the head to point to the new node */
    (*head_ref) = new_node;
}

Node* addTwoLists(Node* first, Node* second)
{
    // res is head node of the resultant list
    Node* res = NULL;
    Node *temp, *prev = NULL;
    int carry = 0, sum;
 
    // while both lists exist
    while (first != NULL || second != NULL) {
        // Calculate value of next digit in resultant list.
        // The next digit is sum of following things
          // (i) Carry
        // (ii) Next digit of first list (if there is a next digit)
        // (ii) Next digit of second list (if there is a next digit)
        sum = carry + (first ? first->data : 0) + (second ? second->data : 0);
        // update carry for next calculation
        carry = (sum >= 10) ? 1 : 0;
        // update sum if it is greater than 10
        sum = sum % 10;
        // Create a new node with sum as data
        temp = newNode(sum);
        // if this is the first node then set it as head of the resultant list
        if (res == NULL)
            res = temp;
        // If this is not the first node then connect it to the rest.
        else
            prev->next = temp;
       
        // Set prev for next insertion
        prev = temp;
 
        // Move first and second pointers to next nodes
        if (first)
            first = first->next;
        if (second)
            second = second->next;
    }
    if (carry > 0)
        temp->next = newNode(carry);
    // return head of the resultant list
    return res;
}


Add two numbers represented by linked lists | Set 2
======================================================
Given two numbers represented by two linked lists, write a function that 
returns the sum list. The sum list is linked list representation of the 
addition of two input numbers. It is not allowed to modify the lists. Also, 
not allowed to use explicit extra space (Hint: Use Recursion).

Example :

Input:
  First List: 5->6->3  
  Second List: 8->4->2 
Output
  Resultant list: 1->4->0->5


typedef Node node;
// A utility function to swap two pointers
void swapPointer(Node** a, Node** b)
{
    node* t = *a;
    *a = *b;
    *b = t;
}
// Adds two linked lists of same size
// represented by head1 and head2 and returns
// head of the resultant linked list. Carry
// is propagated while returning from the recursion
node* addSameSize(Node* head1, Node* head2, int* carry)
{
    // Since the function assumes linked lists are of same
    // size, check any of the two head pointers
    if (head1 == NULL)
        return NULL;
  
    int sum;
  
    // Allocate memory for sum node of current two nodes
    Node* result = new Node[(sizeof(Node))];
  
    // Recursively add remaining nodes and get the carry
    result->next
        = addSameSize(head1->next, head2->next, carry);
  
    // add digits of current nodes and propagated carry
    sum = head1->data + head2->data + *carry;
    *carry = sum / 10;
    sum = sum % 10;
  
    // Assign the sum to current node of resultant list
    result->data = sum;
  
    return result;
}
  
// This function is called after the
// smaller list is added to the bigger
// lists's sublist of same size. Once the
// right sublist is added, the carry
// must be added toe left side of larger
// list to get the final result.
void addCarryToRemaining(Node* head1, Node* cur, int* carry,
                         Node** result)
{
    int sum;
  
    // If diff. number of nodes are not traversed, add carry
    if (head1 != cur) {
        addCarryToRemaining(head1->next, cur, carry,
                            result);
  
        sum = head1->data + *carry;
        *carry = sum / 10;
        sum %= 10;
  
        // add this node to the front of the result
        push(result, sum);
    }
}
  
// The main function that adds two linked lists
// represented by head1 and head2. The sum of
// two lists is stored in a list referred by result
void addList(Node* head1, Node* head2, Node** result)
{
    Node* cur;
  
    // first list is empty
    if (head1 == NULL) {
        *result = head2;
        return;
    }
  
    // second list is empty
    else if (head2 == NULL) {
        *result = head1;
        return;
    }
  
    int size1 = getSize(head1);
    int size2 = getSize(head2);
  
    int carry = 0;
  
    // Add same size lists
    if (size1 == size2)
        *result = addSameSize(head1, head2, &carry);
  
    else {
        int diff = abs(size1 - size2);
  
        // First list should always be larger than second
        // list. If not, swap pointers
        if (size1 < size2)
            swapPointer(&head1, &head2);
  
        // move diff. number of nodes in first list
        for (cur = head1; diff--; cur = cur->next)
            ;
  
        // get addition of same size lists
        *result = addSameSize(cur, head2, &carry);
  
        // get addition of remaining first list and carry
        addCarryToRemaining(head1, cur, &carry, result);
    }
  
    // if some carry is still there, add a new node to the
    // front of the result list. e.g. 999 and 87
    if (carry)
        push(result, carry);
}

Multiply two numbers represented as linked lists into a third list
=====================================================================
Given two numbers represented by linked lists, write a function that returns 
the head of the new linked list that represents the number that is the product 
of those numbers.

Examples: 

Input : 9->4->6
        8->4
Output : 7->9->4->6->4

Input : 9->9->9->4->6->9
        9->9->8->4->9
Output : 9->9->7->9->5->9->8->0->1->8->1

struct Node* multiplyTwoLists(struct Node* first,struct Node* second)
{
    // reverse the lists to multiply from end
    // m and n lengths of linked lists to make
    // and empty list
    int m = reverse(&first), n = reverse(&second);
 
    // make a list that will contain the result
    // of multiplication.
    // m+n+1 can be max size of the list
    struct Node* result = make_empty_list(m + n + 1);
 
    // pointers for traverse linked lists and also
    // to reverse them after
    struct Node *second_ptr = second,*result_ptr1 = result, *result_ptr2, *first_ptr;
 
    // multiply each Node of second list with first
    while (second_ptr) {
 
        int carry = 0;
 
        // each time we start from the next of Node
        // from which we started last time
        result_ptr2 = result_ptr1;
 
        first_ptr = first;
 
        while (first_ptr) {
 
            // multiply a first list's digit with a
            // current second list's digit
            int mul = first_ptr->data * second_ptr->data + carry;
 
            // Assign the product to corresponding Node
            // of result
            result_ptr2->data += mul % 10;
 
            // now resultant Node itself can have more
            // than 1 digit
            carry = mul / 10 + result_ptr2->data / 10;
            result_ptr2->data = result_ptr2->data % 10;
 
            first_ptr = first_ptr->next;
            result_ptr2 = result_ptr2->next;
        }
 
        // if carry is remaining from last multiplication
        if (carry > 0) {
            result_ptr2->data += carry;
        }
 
        result_ptr1 = result_ptr1->next;
        second_ptr = second_ptr->next;
    }
 
    // reverse the result_list as it was populated
    // from last Node
    reverse(&result);
    reverse(&first);
    reverse(&second);
 
    // remove if there are zeros at starting
    while (result->data == 0) {
        struct Node* temp = result;
        result = result->next;
        free(temp);
    }
 
    // Return head of multiplication list
    return result;
}


