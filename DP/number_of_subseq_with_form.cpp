Number of subsequences of the form a^i b^j c^k
===============================================
// Given a string, count number of subsequences of the form aibjck, i.e., 
// it consists of i ’a’ characters, followed by j ’b’ characters, followed 
// by k ’c’ characters where i >= 1, j >=1 and k >= 1. 

// Note: Two subsequences are considered different if the set of array indexes 
// picked for the 2 subsequences are different.

// Expected Time Complexity: O(n)

// Examples: 

// Input  : abbc
// Output : 3
// Subsequences are abc, abc and abbc

// Input  : abcabc
// Output : 7
// Subsequences are abc, abc, abbc, aabc
// abcc, abc and abc


// Approach: 

// We traverse given string. For every character encounter, we do the following:

// Initialize counts of different subsequences caused by different combination 
// of ‘a’. Let this count be aCount.
// Initialize counts of different subsequences caused by different combination 
// of ‘b’. Let this count be bCount.
// Initialize counts of different subsequences caused by different combination 
// of ‘c’. Let this count be cCount.
// Traverse all characters of given string. Do following for current character s[i] 
//         If current character is ‘a’, then there are following possibilities :O(1). 
//         Current character begins a new subsequence. 
//         Current character is part of aCount subsequences. 
//         Current character is not part of aCount subsequences. 
//         Therefore we do aCount = (1 + 2 * aCount);
//     If current character is ‘b’, then there are following possibilities : 
//         Current character begins a new subsequence of b’s with aCount subsequences. 
//         Current character is part of bCount subsequences. 
//         Current character is not part of bCount subsequences. 
//         Therefore we do bCount = (aCount + 2 * bCount);
//     If current character is ‘c’, then there are following possibilities : 
//         Current character begins a new subsequence of c’s with bCount subsequences. 
//         Current character is part of cCount subsequences. 
//         Current character is not part of cCount subsequences. 
//         Therefore we do cCount = (bCount + 2 * cCount);
// Finally we return cCount;

// Explanation of approach with help of example: 

// aCount is the number of subsequences of the letter ‘a’.

// Consider this example: aa.

// We can see that aCount for this is 3, because we can choose these 
// possibilities: (xa, ax, aa) (x means we did not use that character). Note also 
// that this is independent of characters in between, i.e. the aCount of aa and 
// ccbabbbcac are the same because both have exactly 2 a’s.

// Now, adding 1 a, we now have the following new subsequences: each of the 
// old subsequences, each of the old subsequences + the new a, and the new letter a, 
// alone. So a total of aCount + aCount + 1 subsequences.

// Now, let’s consider bCount, the number of subsequences with some a’s and then 
// some b’s. in ‘aab’, we see that bCount should be 3 (axb, xab, aab) because it 
// is just the number of ways we can choose subsequences of the first two a’s, and 
// then b. So every time we add a b, the number of ways increases by aCount.

// Let’s find bCount for ‘aabb’. We have already determined that aab has 3 
// subsequences, so certainly we still have those. Additionally, we can add the 
// new b onto any of these subsequences, to get 3 more. Finally, we have to count 
// the subsequences that are made without using any other b’s, and by the logic 
// in the last paragraph, that is just aCount. So, bCount after this is just the 
// old bCount*2 + aCount;

// cCount is similar. 

// Returns count of subsequences of the form
// a^i b^j c^k
int countSubsequences(string s)
{
    // Initialize counts of different subsequences
    // caused by different combination of 'a'
    int aCount = 0;
 
    // Initialize counts of different subsequences
    // caused by different combination of 'a' and
    // different combination of 'b'
    int bCount = 0;
 
    // Initialize counts of different subsequences
    // caused by different combination of 'a', 'b'
    // and 'c'.
    int cCount = 0;
 
    // Traverse all characters of given string
    for (unsigned int i = 0; i < s.size(); i++) {
        /* If current character is 'a', then
           there are the following possibilities :
             a) Current character begins a new
                subsequence.
             b) Current character is part of aCount
                subsequences.
             c) Current character is not part of
                aCount subsequences. */
        if (s[i] == 'a')
            aCount = (1 + 2 * aCount);
 
        /* If current character is 'b', then
           there are following possibilities :
             a) Current character begins a new
                subsequence of b's with aCount
                subsequences.
             b) Current character is part of bCount
                subsequences.
             c) Current character is not part of
                bCount subsequences. */
        else if (s[i] == 'b')
            bCount = (aCount + 2 * bCount);
 
        /* If current character is 'c', then
           there are following possibilities :
             a) Current character begins a new
                subsequence of c's with bCount
                subsequences.
             b) Current character is part of cCount
                subsequences.
             c) Current character is not part of
                cCount subsequences. */
        else if (s[i] == 'c')
            cCount = (bCount + 2 * cCount);
    }
 
    return cCount;
}

// Output

// 3

// Complexity Analysis: 

//     Time Complexity: O(n). 
//     One traversal of the string is needed.
//     Auxiliary Space: O(1). 
//     No extra space is needed.