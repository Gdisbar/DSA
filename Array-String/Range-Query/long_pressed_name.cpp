925. Long Pressed Name
========================
// Your friend is typing his name into a keyboard. Sometimes, when typing a character c, 
// the key might get long pressed, and the character will be typed 1 or more times.

// You examine the typed characters of the keyboard. Return True if it is possible 
// that it was your friends name, with some characters (possibly none) being long pressed.

 

// Example 1:

// Input: name = "alex", typed = "aaleex"
// Output: true
// Explanation: 'a' and 'e' in 'alex' were long pressed.

// Example 2:

// Input: name = "saeed", typed = "ssaaedd"
// Output: false
// Explanation: 'e' must have been pressed twice, but it was not in the typed output.

 

// Constraints:

//     1 <= name.length, typed.length <= 1000
//     name and typed consist of only lowercase English letters.


// using map will fail for these test cases :
// "xnhtq"
// "xhhttqq"
// "rick"
// "kric"
// "alex"
// "aaleexa"

    bool isLongPressedName(string name, string typed) {
        int i = 0, m = name.length(), n = typed.length();
        for (int j = 0; j < n; ++j)
            if (i < m && name[i] == typed[j])
                ++i;
            else if (!j || typed[j] != typed[j - 1])
                return false;
        return i == m;
    }

class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        if name==typed: return True
        if len(name)>len(typed): return False
        i,j=0,0
        while i<len(name) and j<len(typed):
            a,b=name[i],typed[j]
            if a!=b: return False
            # move both i and j forward if they all match
            while i<len(name) and j<len(typed) and name[i]==a and typed[j]==b:
                i+=1
                j+=1
            #move j only if i in name has enough match
            while j<len(typed) and typed[j]==b: j+=1
        # process unmatched char in name and typed
        if i==len(name) and j<len(typed) or j==len(typed) and i<len(name): return False
        return True