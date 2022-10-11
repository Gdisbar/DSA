670. Maximum Swap
====================
// You are given an integer num. You can swap two digits at most once to get 
// the maximum valued number.

// Return the maximum valued number you can get.

 

// Example 1:

// Input: num = 2736
// Output: 7236
// Explanation: Swap the number 2 and the number 7.

// Example 2:

// Input: num = 9973
// Output: 9973
// Explanation: No swap.

//If you just think about the conditions there are lots of it 
//Brute Force : use 2 loops to swap & check

//Find rightmx of each number starting from right to left ---> TC : n
// i= 0 1 2 3 4 5 6 7 8 9
// a= 9 9 8 7 4 3 6 2 9 4
// r= 8 8 8 8 8 8 8 8 8 9
//a[i]!=a[r[i]] ---> element at index i is equal to element at r[i] , if not swap
//here 9(a[i=0])==9(r[i=0]=8),9(a[i=1])==9(a[r[i=1]=8]),8(a[i=2])!=9(a[r[i=2]=8])--> swap
//so we check a[i]!=a[r[i]] --> swapping condition

// TC : n , sc : 10 , 100% faster,60% less memory

int maximumSwap(int num) {
        string s=to_string(num);
        vector<int> idx(10,-1);
        for(int i=0;i<s.size();++i)
            idx[s[i]-'0']=i;
        bool flag=false;
        for(int i=0;i<s.size();++i){
            int digit=s[i]-'0';
            for(int j=9;j>digit;--j){
                if(i<idx[j]){
                    
                    swap(s[i],s[idx[j]]);
                    flag=true;
                    break;
                }
            }
            if(flag) break;
        }
        return stoi(s);
    }


// Recursive solution

void helper(string a, int start){
        if(start==a.size()) return;

        // try to find a number greater than the current
        // note, if there are several max numbers we need to take the last one,
        // e.g. 1993->9913 rather than 9193
        int maxIdx=start;
        for(int i=start+1;i<a.size();i++){
            if(a[i]>a[start] && a[i]>=a[maxIdx]) maxIdx = i;
        }

        // if max was found
        if(maxIdx>start){
        	swap(a[maxIdx],a[start]);
            return;
        }

        helper(a, start+1);
    }

int maximumSwap(int num) {
        string a = to_string(num);
        helper(a, 0);
        return stoi(a);
    }

    