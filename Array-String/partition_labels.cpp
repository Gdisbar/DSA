763. Partition Labels
======================
You are given a string s. We want to partition the string into as many parts as 
possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, 
the resultant string should be s.

Return a list of integers representing the size of these parts.

 

Example 1:

Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s 
into less parts.

Example 2:

Input: s = "eccbbbbdec"
Output: [10]

Example 3:
Input: s = "caedbdedda"
//Output: [10]  
Expected: [1,9]

if we initalize h=mp[s[0]] & start loop from i=1 ,1st char apperared only in i=0 
but we''re checking from  i=1 so we don''t make separate partition 
for 'c' --> which is wrong

//TC : n , SC : 26
vector<int> partitionLabels(string s) {
        int n = s.size();
        vector<int> v;
        v.clear();
        vector<int> mp(26,0); //array takes less time than unordered_map
        mp.clear();
        for(int i = 0;i<n;i++){
           mp[s[i]-'a']=i;
        }
        int l=0,h=0;
        for(int i = 0;i<n;i++){
          if(mp[s[i]-'a']>h) h=mp[s[i]-'a'];
          if(i==h){ 
            v.push_back(h-l+1);
            l=h+1;
          }
        }
        return v;
    }

// just to get familiar with python , exact same code

// class Solution:
//     def partitionLabels(self, s: str) -> List[int]:
//         L = len(s)
//         last = {s[i]: i for i in range(L)} # last appearance of the letter
//         i, ans = 0, []
//         while i < L:
//             end, j = last[s[i]], i + 1
//             while j < end: # validation of the part [i, end]
//                 if last[s[j]] > end:
//                     end = last[s[j]] # extend the part
//                 j += 1
           
//             ans.append(end - i + 1)
//             i = end + 1
            
//         return ans