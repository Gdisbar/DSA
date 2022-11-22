402. Remove K Digits
======================
// Given string num representing a non-negative integer num, and an integer k, 
// return the smallest possible integer after removing k digits from num.


// Example 1:

// Input: num = "1432219", k = 3
// Output: "1219"
// Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 
// which is the smallest.

st = 4,1 -> 1 k=2
st = 3,1 -> 1 k=1
st = 2,1 -> k=0 -> st= 9,1,2,1
num=143|1219 , n=3

// Example 2:

// Input: num = "10200", k = 1
// Output: "200"
// Explanation: Remove the leading 1 and the number is 200. Note that the 
// output must not contain leading zeroes.

// Example 3:

// Input: num = "10", k = 2
// Output: "0"
// Explanation: Remove all the digits from the number and it is left with nothing 
// which is 0.


string removeKdigits(string num, int k) {
	// number of operation greater than length we return an empty string
    if(num.length() <= k)  return "0";
    // k is 0 , no need of removing /  preforming any operation
    if(k == 0) return num;
    stack<char> st;
    //st.push(num[0]);
    int n=num.size();
    for(int i=0;i<n;++i){
            while(!st.empty()&&st.top()>num[i]&&k>0){
                st.pop();
                k--;
            }
        if(!st.empty()||num[i]!='0') //prevent leading zeros
            st.push(num[i]);
        //if(k==0) break;
        
    }
    while(!st.empty()&&k>0){ // k not fully spent 
    // for cases like "456" where every num[i] > num.top()
        k--;
        st.pop();
    }
    if(st.empty()) return "0";
    while(!st.empty()){
        num[n-1]=st.top();
        st.pop();
        n--;
    }
    
    return num.substr(n);
}

def removeKdigits(self, num: str, k: int) -> str:
        st = list()
        for n in num:
            while st and k and st[-1] > n:
                st.pop()
                k -= 1
            
            if st or n is not '0': # prevent leading zeros
                st.append(n)
                
        if k: # not fully spent
			st = st[0:-k]
            
        return ''.join(st) or '0'