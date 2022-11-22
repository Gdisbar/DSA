Huffman Encoding
=====================   
// Given a string S of distinct character of size N and their corresponding 
// frequency f[ ] i.e. character S[i] has f[i] frequency. Your task is to build 
// the Huffman tree print all the huffman codes in preorder traversal of the tree.
// Note: While merging if two nodes have the same value, then the node which 
// occurs at first will be taken on the left of Binary Tree and the other one to 
// the right, otherwise Node with less value will be taken on the left of the 
// subtree and other one to the right.
						

// 						   100
// 					   0  /   \ 1
// 					     45    55
// 					      0 /    \ 1
// 					       25     30
// 					    0 / \ 1  0/ \ 1
// 					     12 13  14   16
// 					          0 / \ 1
// 					           5   9 
// Example 1:

// S = "abcdef"
// f[] = {5, 9, 12, 13, 16, 45}
// Output: 
// 0 100 101 1100 1101 111
// Explanation:
// Steps to print codes from Huffman Tree
// HuffmanCodes will be:
// f : 0
// c : 100
// d : 101
// a : 1100
// b : 1101
// e : 111
// Hence printing them in the PreOrder of Binary 
// Tree.



class Solution
{
	public:
	struct node
	{
	    node* left,*right;
	    int val;
	    node(int vals)
	    {
	        val=vals;
	        left=right=NULL;
	    }
	};
	
	struct cmp
	{
	    bool operator()(const node* a,const node* b)
	    {
	        return (a->val>b->val);
	    }
	};
	    void preo(node* root,string a,vector<string>&v)
	    {
	        if(root==NULL){
	            return ;
	        }
	        if(root->left==NULL&&root->right==NULL)
	        {
	           // a+=(root->val);
	            v.push_back(a); //reached leaf node recurssion for this subtree is over
	            return ;
	        }
	        preo(root->left,a+"0",v);
	        preo(root->right,a+"1",v);
	    }
		vector<string> huffmanCodes(string S,vector<int> f,int N)
		{
		    priority_queue<node*,vector<node*>,cmp>pq; //min heap
		    
		    for(int i=0;i<N;i++)
		    {
		        pq.push(new node(f[i]));
		    }
		    vector<string>v;
		    
		    while(pq.size()!=1)
		    {
		        node* l=pq.top(); //take 1st lowest in pq
		        pq.pop();
		        node* r=pq.top(); // take 2nd lowest in pq
		        pq.pop();

		        // create new node with sum of l & r
		        node* par=new node(l->val+r->val); 
		        par->left=l;
		        par->right=r;
		        
		        pq.push(par);
		    }
		    string a="";
		    node* root=pq.top();
		    pq.pop();
		    preo(root,"",v);
		    
		    return v;
		}
};

// { Driver Code Starts.
int main(){
    int T;
    cin >> T;
    while(T--)
    {
	    string S;
	    cin >> S;
	    int N = S.length();
	    vector<int> f(N);
	    for(int i=0;i<N;i++){
	        cin>>f[i];
	    }
	    Solution ob;
	    vector<string> ans = ob.huffmanCodes(S,f,N);
	    for(auto i: ans)
	    	cout << i << " ";
	    cout << "\n";
    }
	return 0;
}