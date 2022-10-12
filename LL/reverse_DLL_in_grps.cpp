Reverse a doubly linked list in groups of given size
=========================================================
Given a doubly linked list containing n nodes. 
The problem is to reverse every group of k nodes in the list.

// TC : n , Recursive

struct Node {
    int data;
    Node *next, *prev;
};
// function to add Node at the end of a Doubly LinkedList
Node* insertAtEnd(Node* head, int data)
{
  
    Node* new_node = new Node();
    new_node->data = data;
    new_node->next = NULL;
    Node* temp = head;
    if (head == NULL) {
        new_node->prev = NULL;
        head = new_node;
        return head;
    }
  
    while (temp->next != NULL) {
        temp = temp->next;
    }
    temp->next = new_node;
    new_node->prev = temp;
    return head;
}
// function to print Doubly LinkedList
void printDLL(Node* head)
{
    while (head != NULL) {
        cout << head->data << " ";
        head = head->next;
    }
    cout << endl;
}
// function to Reverse a doubly linked list
// in groups of given size
Node* reverseByN(Node* head, int k)
{
    if (!head)
        return NULL;
    head->prev = NULL;
    Node *temp, *curr = head, *newHead;
    int count = 0;
    while (curr != NULL && count < k) {
        newHead = curr;
        temp = curr->prev;
        curr->prev = curr->next;
        curr->next = temp;
        curr = curr->prev;
        count++;
    }
    // checking if the reversed LinkedList size is
    // equal to K or not
    // if it is not equal to k that means we have reversed
    // the last set of size K and we don't need to call the
    // recursive function
    if (count >= k) {
        Node* rest = reverseByN(curr, k);
        head->next = rest;
        if (rest != NULL)
            // it is required for prev link otherwise u wont
            // be backtrack list due to broken links
            rest->prev = head;
    }
    return newHead;
}
int main()
{
    Node* head;
    for (int i = 1; i <= 10; i++) {
        head = insertAtEnd(head, i);
    }
    printDLL(head);
    int n = 4;
    head = reverseByN(head, n);
    printDLL(head);
}



Original list: 10 8 4 2 
Modified list: 8 10 2 4 


// iterative 

// function to get a new node
Node* getNode(int data)
{
    // allocating node
    Node* new_node = new Node();
    new_node->data = data;
    new_node->next = new_node->prev = NULL;
  
    return new_node;
}
  
// function to insert a node at the beginning
// of the Doubly Linked List
Node* push(Node* head, Node* new_node)
{
    // since we are adding at the beginning,
    // prev is always NULL
    new_node->prev = NULL;
  
    // link the old list off the new node
    new_node->next = head;
    // change prev of head node to new node
    if (head != NULL)
        head->prev = new_node;
  
    // move the head to point to the new node
    head = new_node;
    return head;
}
  
// function to reverse a doubly linked list
// in groups of given size
Node* revListInGroupOfGivenSize(Node* head, int k)
{
    if (!head)
        return head;
  
    Node* st = head;
    Node* globprev = NULL;
    Node* ans = NULL;
    while (st) {
        int count = 1; // to count k nodes
        Node* curr = st;
        Node* prev = NULL;
        Node* next = NULL;
        while (curr && count <= k) { // reversing k nodes
            next = curr->next;
            curr->prev = next;
            curr->next = prev;
            prev = curr;
            curr = next;
            count++;
        }
  
        if (!ans) {
            ans = prev; // to store ans i.e the new head
            ans->prev = NULL;
        }
  
        if (!globprev) globprev = st; // assigning the last node of the
                           // reversed k nodes
        else {
            globprev->next = prev;
            prev->prev = globprev; // connecting last node of last
                            // k group to the first node of
                            // present k group
            globprev = st;
        }
  
        st = curr; // advancing the pointer for the next k
                   // group
    }
    return ans;
}
