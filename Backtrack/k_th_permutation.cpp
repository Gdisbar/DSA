60. Permutation Sequence
===============================================================
// The set [1, 2, 3, ..., n] contains a total of n! unique permutations.

// By listing and labeling all of the permutations in order, we get the 
// following sequence for n = 3:

//     "123"
//     "132"
//     "213"
//     "231"
//     "312"
//     "321"

// Given n and k, return the kth permutation sequence.

 

// Example 1:

// Input: n = 3, k = 3
// Output: "213"

// Example 2:

// Input: n = 4, k = 9
// Output: "2314"

// Example 3:

// Input: n = 3, k = 1
// Output: "123"
 
// for n=4 k = 15, we have 4!=24 possible permutations

// arr
// [ 1            2           3          4]
// 1 2 3 4     2 1 3 4     3 1 2 4    4 1 2 3
// 1 2 4 3     2 1 4 3     3 1 4 2    4 1 3 2
// 1 3 2 4     2 3 1 4     3 2 1 4    4 2 1 3
// 1 3 4 2     2 3 4 1     3 2 4 1    4 2 3 1
// 1 4 2 3     2 4 1 3     3 4 1 2    4 3 1 2
// 1 4 3 2     2 4 3 1     3 4 2 1    4 3 2 1
// So we have 4 block with n-1! = 3! = 6 elements each ---> 15 = 6*2 + 3  
// i.e. we skip 2 blocks and our ans is the third element in the 3rd block
// 15 / 6 = 2;  So we select the block at 2nd index i.e block 3 , ans=3

// Now we have 3 blocks each of with n-1!= 2!=2 elements 
// 3 1 2 4  - 1  
// 3 1 4 2  - 2   Block 0
//   ------ 
// 3 2 1 4  - 3 (ans)
// 3 2 4 1  - 4    Block 1
//   ------
// 3 4 1 2  - 5   
// 3 4 2 1  - 6    Block 2

// k / p = 3 / 2 = 1 => ans is in block 1, ans=ans(3)+2=32

// Now we have 2 blocks each of with n-1!= 1!=1 elements 
// 3 2 1 4  - 3    Block 0
// 3 2 4 1  - 4    Block 1

// ans=ans(1)+32 = 321
// As we only have one value value in array append it to ans.  ans = "3214"


// One very important note:(Corner case)
// When we have k as a multiple of elements in partition for e.g. k = 12 
// Then we want to be in block with index 1
// but as index = 12 / 6 = 2; we have to keep index = index-1;
// Only when we are aiming at the last element we will hit this case.
// Here the blocks are zero indexed but the elements inside them are 1 index.


// factVal : factorial of 0-9
void setPerm(vector<int>& v,string& ans,int n,int k,vector<int>& factVal){
       // if there is only one element left append it to our ans (Base case)
       if(n==1){
            ans+=to_string(v.back());
            return;
        }
        
        // We are calculating the required index.  factVal[n-1] means for 
        // n =  4 => factVal[3] = 6.
        // 15 / 6 = 2 will the index for k =15 and n = 4.
        int index = (k/factVal[n-1]);
        // if k is a multiple of elements of partition then decrement the 
        // index (Corner case)
        if(k % factVal[n-1] == 0){
            index--;
        }
        
        ans+= to_string(v[index]);  // add value to string
        v.erase(v.begin() + index);  // remove element from array
        k -= factVal[n-1] * index;   // adjust value of k; k = 15 - 6*2 = 3.
        // Recursive call with n=n-1 as one element is added we need remaing.
        setPerm(v,ans,n-1,k,factVal);
    }
    
    string getPermutation(int n, int k) {
        if(n==1) return "1";
        //Factorials of 0-9 stored in the array. factVal[3] = 6. (3! = 6)
        vector<int>factVal = {1,1,2,6,24,120,720,5040,40320,362880};
        string ans = "";
        vector<int> v;
        for(int i=1;i<=n;i++) v.emplace_back(i);
        setPerm(v,ans,n,k,factVal);
        return ans;
    }