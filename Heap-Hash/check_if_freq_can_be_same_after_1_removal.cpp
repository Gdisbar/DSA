Check if frequency of all characters can become same by one removal
======================================================================
// Given a string which contains lower alphabetic characters, 
// we need to remove at most one character from this string in such a way 
// that frequency of each distinct character becomes same in the string.

// Examples:  

//     Input: str = “xyyz” 
//     Output: Yes 
//     We can remove character ’y’ from above 
//     string to make the frequency of each 
//     character same. 

//     Input: str = “xyyzz” 
//     Output: Yes 
//     We can remove character ‘x’ from above 
//     string to make the frequency of each 
//     character same.

//     Input: str = “xxxxyyzz” 
//     Output: No 
//     It is not possible to make frequency of 
//     each character same just by removing at 
//     most one character from above string.


#define M 26
 
// Utility method to get index of character ch
// in lower alphabet characters
int getIdx(char ch)
{
    return (ch - 'a');
}
 
// Returns true if all non-zero elements
// values are same
bool allSame(int freq[], int N)
{
    int same;
 
    // get first non-zero element
    int i;
    for (i = 0; i < N; i++) {
        if (freq[i] > 0) {
            same = freq[i];
            break;
        }
    }
 
    // check equality of each element with variable same
    for (int j = i + 1; j < N; j++)
        if (freq[j] > 0 && freq[j] != same)
            return false;
 
    return true;
}
 
// Returns true if we can make all character
// frequencies same
bool possibleSameCharFreqByOneRemoval(string str)
{
    int l = str.length();
 
    // fill frequency array
    int freq[M] = { 0 };
    for (int i = 0; i < l; i++)
        freq[getIdx(str[i])]++;
 
    // if all frequencies are same, then return true
    if (allSame(freq, M))
        return true;
 
    /*  Try decreasing frequency of all character
        by one and then    check all equality of all
        non-zero frequencies */
    for (char c = 'a'; c <= 'z'; c++) {
        int i = getIdx(c);
 
        // Check character only if it occurs in str
        if (freq[i] > 0) {
            freq[i]--;
 
            if (allSame(freq, M))
                return true;
            freq[i]++;
        }
    }
 
    return false;
}
 