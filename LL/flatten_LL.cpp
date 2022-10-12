Flattening a Linked List 
===============================
Given a Linked List of size N, where every node represents a sub-linked-list and contains two pointers:
(i) a next pointer to the next node,
(ii) a bottom pointer to a linked list where this node is head.
Each of the sub-linked-list is in sorted order.
Flatten the Link List such that all the nodes appear in a single level while maintaining the sorted order. 
Note: The flattened list will be printed using the bottom pointer instead of next pointer.

 

Example 1:

Input:
5 -> 10 -> 19 -> 28
|     |     |     | 
7     20    22   35
|           |     | 
8          50    40
|                 | 
30               45
Output:  5-> 7-> 8- > 10 -> 19-> 20->
22-> 28-> 30-> 35-> 40-> 45-> 50.
Explanation:
The resultant linked lists has every 
node in a single level.
(Note: | represents the bottom pointer.)

 

Example 2:

Input:
5 -> 10 -> 19 -> 28
|          |                
7          22   
|          |                 
8          50 
|                           
30              
Output: 5->7->8->10->19->22->28->30->50
Explanation:
The resultant linked lists has every
node in a single level.

(Note: | represents the bottom pointer.)


/* Node structure  used in the program

struct Node{
	int data;
	struct Node * next;
	struct Node * bottom;
	
	Node(int x){
	    data = x;
	    next = NULL;
	    bottom = NULL;
	}
	
};
*/
// TC : n*m
Node* merge(Node* a,Node* b){
    if(!a) return b;
    if(!b) return a;
    Node* head;
    if(a->data<b->data){
        head=a;
        head->bottom=merge(a->bottom,b);
    }
    else{
        head=b;
        head->bottom=merge(a,b->bottom);
    }
    head->next=nullptr; //only bottom exist, no next as we're comparing with last merged list
    return head;
}
/*  Function which returns the  root of 
    the flattened linked list. */
Node *flatten(Node *root)
{
   // Your code here
   if(!root or !root->next) return root;
   return merge(root,flatten(root->next)); //start merging from last
}

// TC : n*m*log(n) , using Priority Queue

struct mycomp {
    bool operator()(Node* a, Node* b)
    {
        return a->data > b->data;
    }
};
void flatten(Node* root)
{
    priority_queue<Node*, vector<Node*>, mycomp> p;
  //pushing main link nodes into priority_queue.
    while (root!=NULL) {
        p.push(root);
        root = root->next;
    }
   
    while (!p.empty()) {
      //extracting min
        auto k = p.top();
        p.pop();
      //printing  least element
        cout << k->data << " ";
        if (k->bottom)
            p.push(k->bottom);
    }
    
}

430. Flatten a Multilevel Doubly Linked List
==================================================
You are given a doubly linked list, which contains nodes that have a next pointer, 
a previous pointer, and an additional child pointer. This child pointer may or 
may not point to a separate doubly linked list, also containing these special nodes. 
These child lists may have one or more children of their own, and so on, to produce 
a multilevel data structure as shown in the example below.

Given the head of the first level of the list, flatten the list so that all the 
nodes appear in a single-level, doubly linked list. Let curr be a node with a 
child list. The nodes in the child list should appear after curr and before 
curr.next in the flattened list.

Return the head of the flattened list. The nodes in the list must have all of their 
child pointers set to null.

 1 -> 2 -> 3 -> 4 -> 5 -> 6
           |
           7 -> 8 -> 9 -> 10
                |
                11 -> 12
Example 1:

Input: head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
Output: [1,2,3,7,8,11,12,9,10,4,5,6]
Explanation: The multilevel linked list in the input is shown.
After flattening the multilevel linked list it becomes:

Example 2:

Input: head = [1,2,null,3]
Output: [1,3,2]
Explanation: The multilevel linked list in the input is shown.
After flattening the multilevel linked list it becomes:

Example 3:

Input: head = []
Output: []
Explanation: There could be empty list in the input.

// TC : n*n

lets say we start out with :

h    
1 - 2 - 3 - 4 - 5 - null
    |
    6 - 7 - 8 - null
            |
            9 - 10 - null

h points to the head of the structure
horizontal links are bidirectional
vertical links indicate child relationship

after the first child is encountered:

        h
1 - 2 - 6 - 7 - 8 - 3 - 4 - 5 - null
                |
                9 - 10 - null

after the second child is encountered:

                    h
1 - 2 - 6 - 7 - 8 - 9 - 10 - 3 - 4 - 5 - null


 
Node* flatten(Node* head) {
    for (Node* h = head; h; h = h->next)
    {
        if (h->child)
        {
            Node* next = h->next;
            h->next = h->child;
            h->next->prev = h;
            h->child = NULL;
            Node* p = h->next;
            while (p->next) p = p->next;
            p->next = next;
            if (next) next->prev = p;
        }
    }
    return head;
}


// recursive

Node* flatten(Node* head, Node* rest = nullptr) {
  if (!head) return rest;
  head->next = flatten(head->child, flatten(head->next, rest));
  if (head->next) head->next->prev = head;
  head->child = nullptr;
  return head;
}

This function modifies the structure in place. It''s not the fastest implementation 
out there, but I love short recursive algorithms, and I thought this was rather nice.

The trick to make this work is to add a second parameter to the function signature. 
A call to flatten(head, rest) will flatten head and concatenate rest to the end of it. That allows our recursive definition:

head->next = flatten(head->child, flatten(head->next, rest));

image

(The first line of code is a simple base-case. The third and fourth lines are 
just pointer-cleanup.)

What we''re passing to rest is an already flattened head->next, in order to 
concatenate it to the end of a flattened head->child.
