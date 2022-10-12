//Variable_Size_SlidingWindow
void solve() {
   int a[] = {3,2,4,5,1, 1, 1, 1,1,1,3, 2};
   //int a[] = {3,2,4,5,1, 1, 1, 1,1,1,2, 2};
   int n = sizeof(a)/sizeof(a[0]);
   int k = 5;
   int s = k,w=0;
   
   for (int i = 0; i < n;i++)
   {
      s = s - a[i];
      w++;
      if(s<0){
         if(a[i]==k)
            cout<<"At a["<<i<<"] = "<<a[i]<<" Window size "<<1<<endl; 
         else{
           // w = 0;
            s = k;
            s = s - a[i];
            w = 1;
         }

      }
      else
         cout<<"At a["<<i<<"] = "<<a[i]<<" Window size "<<w<<endl; 
      
   }  
}


//Fixed size Sliding Window

239. Sliding Window Maximum
===============================
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

 

Example 1:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Example 2:

Input: nums = [1], k = 1
Output: [1]

// Example 5:

// Input: nums = [4,-2], k = 2
// Output: [4]



// At each i, we keep "promising" elements, which are potentially max number 
//in window [i-(k-1),i] or any subsequent window. This means

//If an element in the deque and it is out of i-(k-1), we discard them. 
// We just need to poll from the head, as we are using a deque and elements are 
// ordered as the sequence in the array

//Now only those elements within [i-(k-1),i] are in the deque. We then discard 
// elements smaller than a[i] from the tail. This is because if a[x] <a[i] and x<i, 
// then a[x] has no chance to be the "max" in [i-(k-1),i], or any other subsequent 
// window: a[i] would always be a better candidate.

// As a result elements in the deque are ordered in both sequence in array 
//and their value. At each step the head of the deque is the max element in 
//[i-(k-1),i]


//================================================================================================



//Using Deque : O(n),S(k)

    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
// store index, we are inserting,removing & keeping track of elements 
//in the current window,i.e why using dq 
        deque<int> dq;
        if(nums.size()==0||k<=0)
            return ans;
        
        for(int i = 0;i<nums.size();i++){
         // remove numbers out of range k
            while(!dq.empty()&&dq.front()<i-k+1)
                dq.pop_front();
             // remove smaller numbers in k range as they are useless
            while(!dq.empty()&&nums[dq.back()]<nums[i])
                dq.pop_back();
             // q contains index... ans contains content
            dq.push_back(i);
            if(i>=k-1)
                ans.push_back(nums[dq.front()]);
        }
         return ans;       
    }

//================================================================================================
// first -ve number in window
//================================================================================================
arr[] = {5, -2, 3, 4, -5}
k = 2

output = [-2, -2, 0, -5]

vector<long long> printFirstNegativeInteger(long long int A[],
                                             long long int N, long long int K) {
      vector<long long> ans;
      long long j = -1;
      for(long long i = 0;i<N;i++){
          if(j<i){
              do{
                  j++;
              }while(j<N&&A[j]>=0);
          }
          long long x = j<i+K?A[j]:0;
          ans.push_back(x);
      }
      long long r = K-1;
      while(r--){
          ans.pop_back();
      }
      
      return ans;
 }

//Alternate

void maximum (int arr[], int k){
    for (int j = 0; j < arr.size() - k + 1; j++) {
        bool found = false;
        for (int x = j; x < j + k; x++) {
            if (arr[x] < 0) {
                cout<<arr[x]<<" ";
                found = true;
                break;
            }
        }
        if (!found)
            cout<<0<<" ";
    }
}
//================================================================================================
76. Minimum Window Substring
================================
Given two strings s and t of lengths m and n respectively, return the 
minimum window substring of s such that every character in t (including duplicates) 
is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

A substring is a contiguous sequence of characters within the string.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' 
from string t.

Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.

//======================================================================
//  Minimum window substring of s such that every character in t 
//(including duplicates) is included in the window. 

string minWindow(string s, string t) {
        vector<int> map(128,0);
        for(auto c: t) map[c]++;
        int counter=t.size(), begin=0, end=0, d=INT_MAX, head=0;
        while(end<s.size()){
            if(map[s[end++]]-->0) counter--; //got a match in s 
            while(counter==0){ //we've found all characters of t in s
                if(end-begin<d)  d=end-(head=begin);//head store latest begin
                if(map[s[begin++]]++==0) counter++;  //make it invalid
            }  
        }
        return d==INT_MAX? "":s.substr(head, d);
    }
//For most substring problem, we are given a string and need to find a 
// substring of it which satisfy some restrictions. A general way is to use a 
// hashmap assisted with two pointers. The template is given below.

int findSubstring(string s){
        vector<int> map(128,0);
        int counter; // check whether the substring is valid
        int begin=0, end=0; //two pointers, one point to tail and one  head
        int d; //the length of substring

        for() { /* initialize the hash map here */ }

        while(end<s.size()){

            if(map[s[end++]]-- ?){  /* modify counter here */ }

            while(/* counter condition */){ 
                 
                 /* update d here if finding minimum*/

                //increase begin to make it invalid/valid again
                
                if(map[s[begin++]]++ ?){ /*modify counter here*/ }
            }  

            /* update d here if finding maximum*/
        }
        return d;
  }

// One thing needs to be mentioned is that when asked to find maximum substring, 
// we should update maximum after the inner while loop to guarantee that the 
// substring is valid. On the other hand, when asked to find minimum substring, 
// we should update minimum inside the inner while loop.

// The code of solving Longest Substring with At Most Two Distinct Characters 


int lengthOfLongestSubstringTwoDistinct(string s) {
        vector<int> map(128, 0);
        int counter=0, begin=0, end=0, d=0; 
        while(end<s.size()){
            if(map[s[end++]]++==0) counter++;
            while(counter>2) if(map[s[begin++]]--==1) counter--;
            d=max(d, end-begin);
        }
        return d;
    }

//The code of solving Longest Substring Without Repeating Characters is below:


int lengthOfLongestSubstring(string s) {
        vector<int> map(128,0);
        int counter=0, begin=0, end=0, d=0; 
        while(end<s.size()){
            if(map[s[end++]]++>0) counter++; 
            while(counter>0) if(map[s[begin++]]-->1) counter--;
            d=max(d, end-begin); //while valid, update d
        }
        return d;
    }

//Substring with Concatenation of All Words

https://leetcode.com/problems/minimum-window-substring/discuss/26808/Here-is-a-10-line-template-that-can-solve-most-'substring'-problems


//================================================================================================
Find All Anagrams in a String

Input: s = "cbaebabacd", p = "abc"
Output: [0,6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".

Example 2:

Input: s = "abab", p = "ab"
Output: [0,1,2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".





// Now, let's develop an algorithm to solve this problem.

// 1. Find frequency of each character in p.
// 2. Now, we need to find all substrings of length p in s.
//     This process can be efficiently done by using sliding window technique.
//    Sliding Window Technique:-
//    s = abcad, p = abc
//    Take two pointers i and j. 
//    Intially i and j point to starting position of string s. 
//    s = a  b  c  a  d
//         ^
//       i, j
//    ->  move j until j - i == len(p)
//    s = a  b  c  a  d
//         ^        ^
//         i        j
//    Now, the substring formed here is  abc, 
//    it is anagram so, add i to result. and increment i.
//    s = a  b  c  a  d
//            ^     ^
//            i     j
//    Now, j at 3rd index, i at 1st index.
//    3 - 1 < 3
//    so, move j until j - i == len(p)
//    s = a  b  c  a  d
//           ^        ^
//           i        j
//     Now, substring formed here is bca.
//    It is anagram. so, add i to result. and move i.
//    Now, i is at 2nd index and j is at 4th index.
//    4 - 2 < len(p) (3), so move j.
//     s = a  b  c  a  d
//                ^        ^
//                i        j
//    Now, the substring formed here is cad.
//    This is not anagram. Don't do anything.
//    Now, we reached end. So, stop here.
//    This is how we find substring using sliding window technique.
//    and check whether it is anagram or not.
//    if it is anagram, then add starting index to result
// 3. return result.

// TIME:- O(N)
// SPACE:- O(N) -> We are using list to store result


//================================================================================================
    vector<int> findAnagrams(string s, string p) {
        int letters[26] = {0};
        for(char c : p) letters[c - 'a']++;
        
        vector<int> result;
        int remaining = p.size(), j = 0;
        for(int i = 0; i< s.size(); i++){
            while(j < s.size() && j - i < p.size()){
                if(letters[s.at(j++) - 'a']-- > 0)
                    remaining--;
            }
            if(remaining == 0 && j - i == p.size()) 
                result.push_back(i);
            if(letters[s.at(i) - 'a']++ >= 0) 
                remaining++;            
        }
        return result;
    }


