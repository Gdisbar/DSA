844. Backspace String Compare
===============================
// Given two strings s and t, return true if they are equal when both 
// are typed into empty text editors. '#' means a backspace character.

// Note that after backspacing an empty text, the text will continue empty.

 

// Example 1:

// Input: s = "ab#c", t = "ad#c"
// Output: true
// Explanation: Both s and t become "ac".

// Example 2:

// Input: s = "ab##", t = "c#d#"
// Output: true
// Explanation: Both s and t become "".

// Example 3:

// Input: s = "a#c", t = "b"
// Output: false
// Explanation: s becomes "c" while t becomes "b".


// The intuition and quick methode is to find the final text result.
// You can just use a string if you don''t care cost on string modification.
// Or you can use a stack or string builder to do it in O(N).

// Use stack to avoid string modification.
// Time O(N) and space O(N).

    def backspaceCompare(self, S, T):
        def back(res, c):
            if c != '#': res.append(c)
            elif res: res.pop()
            return res
        return reduce(back, S, []) == reduce(back, T, [])


// Follow up: O(1) Space

// Can you do it in O(N) time and O(1) space?
// I believe you have one difficulty here:
// When we meet a char, we are not sure if it will be still there or be deleted.

// However, we can do a back string compare (just like the title of problem).
// If we do it backward, we meet a char and we can be sure this char won't be deleted.
// If we meet a '#', it tell us we need to skip next lowercase char.

// The idea is that, read next letter from end to start.
// If we meet #, we increase the number we need to step back, until back = 0



    bool backspaceCompare(string S, string T) {
        int i = S.length() - 1, j = T.length() - 1, back;
        while (true) {
            back = 0;
            while (i >= 0 && (back > 0 || S[i] == '#')) {
                back += S[i] == '#' ? 1 : -1;
                i--;
            }
            back = 0;
            while (j >= 0 && (back > 0 || T[j] == '#')) {
                back += T[j] == '#' ? 1 : -1;
                j--;
            }
            if (i >= 0 && j >= 0 && S[i] == T[j]) {
                i--;
                j--;
            } else {
                break;
            }
        }
        return i == -1 && j == -1;
    }
