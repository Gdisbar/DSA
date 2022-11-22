106. Construct Binary Tree from Inorder and Postorder Traversal
===================================================================
// Given two integer arrays inorder and postorder where inorder is the inorder 
// traversal of a binary tree and postorder is the postorder traversal 
// of the same tree, construct and return the binary tree.

// Example 1:

// Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
// Output: [3,9,20,null,null,15,7]

// Example 2:

// Input: inorder = [-1], postorder = [-1]
// Output: [-1]

// Then We are going to use Both of the arrays left part to Figur Out Left_subtree
//                      and Both of the arraysRigth Part to Figur out Right_subtree

// We are going to recursively do so until One Of the array dose not got empty

//     inorder   = [4 2 5 1 6 3 7]
//     postorder = [4 5 2 6 7 3 1]

//     So root would be 1 here and Left array which lay left of 1 is [4 2 5] and 
//     Right of 1 is [6 3 7]
//     so left_inorder_array =  [4 2 5] and right_inorder_arry = [6 3 7]

//     using 6 [ which is just rigth of 1] we are going to devide Postorder_array 
//     into two part [4 5 2] and [6 7 3]


//     1st Phase=>        
// 	                   1

//                    /        \

//                 [4 2 5]   [6 3 7]       <= inorder array
//                 [4 5 2]   [6 7 3]       <= postorder array

// Now we have new freash problem like need to make tree by using inorder = [4 2 5] 
// && postorder =  [4 5 2] for left subtree 
// AND inorder = [6 3 7] && postorder = [6 7 3] for right  subtree 
// **now same process we need to do again and again  until One Of the array dose 
// not got empty
// Rest of the Process show in a diagram Form :)

//     2nd Phase =>
//                            1

//                       /        \
//                      2          3
//                 [4]    [5]   [6]   [7]       <= inorder array
//                 [4]    [5]   [6]   [7]       <= postorder array


// 3rd Phase =>  
// 	             1

//                /    \
//               2      3
 
//             /  \    /  \             <==== Answer
 
//            4    5  6    7 


TreeNode *Tree(vector<int>& in, int x, int y,vector<int>& po,int a,int b){
        if(x > y || a > b)return nullptr;
        TreeNode *node = new TreeNode(po[b]);
        int SI = x;  
        while(node->val != in[SI])SI++;
        node->left  = Tree(in,x,SI-1,po,a,a+SI-x-1);
        node->right = Tree(in,SI+1,y,po,a+SI-x,b-1);
        return node;
    }
    TreeNode* buildTree(vector<int>& in, vector<int>& po){
        return Tree(in,0,in.size()-1,po,0,po.size()-1);
    }

unordered_map<int,int>ump;
    TreeNode* build(vector<int>& inorder, vector<int>& postorder,int &rootIdx,int left,int right)
    {
        if(left>right)
        {
            return NULL;
        }
        // int pivot=left;
        // while(inorder[pivot]!=postorder[rootIdx])  ->O(N) search operation eliminated by using unordered-map.
        // {
        //     pivot++;
        // }
        int pivot=ump[postorder[rootIdx]];
        rootIdx--;
        TreeNode* node=new TreeNode(inorder[pivot]);
        
        node->right=build(inorder,postorder,rootIdx,pivot+1,right);
        
        node->left=build(inorder,postorder,rootIdx,left,pivot-1);
        
        return node;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        
        int rootIdx=postorder.size()-1;
        for(int i=0;i<inorder.size();i++)
        {
            ump[inorder[i]]=i;
        }
        
        return build(inorder,postorder, rootIdx, 0,inorder.size()-1);
        
    }


class Solution:
    def buildTree(self, inorder, postorder):
        map_inorder = {}
        for i, val in enumerate(inorder): map_inorder[val] = i
        def recur(low, high):
            if low > high: return None
            x = TreeNode(postorder.pop())
            mid = map_inorder[x.val]
            x.right = recur(mid+1, high)
            x.left = recur(low, mid-1)
            return x
        return recur(0, len(inorder)-1)