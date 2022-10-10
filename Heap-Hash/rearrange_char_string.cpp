Rearrange characters in a String such that no two adjacent characters are same
===============================================================================
// Given a string with lowercase repeated characters, the task is to rearrange 
// characters in a string so that no two adjacent characters are the same. 
// If it is not possible to do so, then print “Not possible”.

// Examples: 

//     Input: aaabc 
//     Output: abaca 

//     Input: aaabb
//     Output: ababa 

//     Input: aa 
//     Output: Not Possible

//     Input: aaaabc 
//     Output: Not Possible


// Build a Priority_queue or max_heap, pq that stores characters with their 
// frequencies. 

//     Priority_queue or max_heap is built on the bases of the frequency of character. 

// Create a temporary Key that will be used as the previously visited element 
// (the previous element in the resultant string. Initialize it 
// 	{ char = ‘#’ , freq = ‘-1’ } 
// While pq is not empty. 

//     Pop an element and add it to the result. 
//     Decrease the frequency of the popped element by ‘1’ 
//     Push the previous element back into the priority_queue if its frequency is 
//     greater than zero. 
//     Make the current element as the previous element for the next iteration. 

// If the length of the resultant string and the original string is not equal, 
// then print “not possible”, else print the resultant string.


const int MAX_CHAR = 26;
 
struct Key {
 
    int freq; // store frequency of character
    char ch;
 
    // Function for priority_queue to store Key
    // according to freq
    bool operator<(const Key& k) const
    {
        return freq < k.freq;
    }
};
 
// Function to rearrange character of a string
// so that no char repeat twice
void rearrangeString(string str)
{
    int N = str.length();
 
    // Store frequencies of all characters in string
    int count[MAX_CHAR] = { 0 };
    for (int i = 0; i < N; i++)
        count[str[i] - 'a']++;
 
    // Insert all characters with their frequencies
    // into a priority_queue
    priority_queue<Key> pq;
    for (char c = 'a'; c <= 'z'; c++) {
        int val = c - 'a';
        if (count[val]) {
            pq.push(Key{ count[val], c });
        }
    }
 
    // 'str' that will store resultant value
    str = "";
 
    // work as the previous visited element
    // initial previous element be. ( '#' and
    // it's frequency '-1' )
    Key prev{ -1, '#' };
 
    // traverse queue
    while (!pq.empty()) {
        // pop top element from queue and add it
        // to string.
        Key k = pq.top();
        pq.pop();
        str = str + k.ch;
 
        // IF frequency of previous character is less
        // than zero that means it is useless, we
        // need not to push it
        if (prev.freq > 0)
            pq.push(prev);
 
        // Make current character as the previous 'char'
        // decrease frequency by 'one'
        (k.freq)--;
        prev = k;
    }
 
    // If length of the resultant string and original
    // string is not same then string is not valid
    if (N != str.length())
        cout << " Not possible " << endl;
 
    else // valid string
        cout << str << endl;
}

# Python program to rearrange characters in a string
# so that no two adjacent characters are same.
 
from heapq import heappush, heappop
from collections import Counter
 
# A key class for readability
 
 
class Key:
    def __init__(self, character: str, freq: int) -> None:
        self.character = character
        self.freq = freq
 
    def __lt__(self, other: "Key") -> bool:
        return self.freq > other.freq
 
 
# Function to rearrange character of a string
# so that no char repeat twice
def rearrangeString(str: str):
    n = len(str)
    # Creating a frequency hashmap
    count = dict()
    for i in str:
        count[ord(i)] = count.get(ord(i), 0) + 1
 
    pq = []
    for c in range(97, 123):
        if count.get(c, 0):
            heappush(pq, Key(chr(c), count))
 
    # null character for default previous checking
    prev = Key('#', -1)
    str = ""
 
    while pq:
        key = heappop(pq)
        str += key.character
 
        # Since one character is already added
        key.freq -= 1
 
        # We avoid inserting if the frequency drops to 0
        if prev.freq > 0:
            heappush(pq, prev)
 
        prev = key
 
    if len(str) != n:
        print("Not possible")
    else:
        print(str)


// Calculate the frequencies of every character in the input string
// If a character with a maximum frequency has a frequency greater than 
// (n + 1) / 2, then return an empty string, as it is not possible to 
// construct a string
// Now fill the even index positions with the maximum frequency character, 
// if some even positions are remaining then first fill them with remaining 
// characters
// Then fill odd index positions with the remaining characters
// Return the constructed string

char getMaxCountChar(vector<int>& count)
{
    int max = 0;
    char ch;
    for (int i = 0; i < 26; i++) {
        if (count[i] > max) {
            max = count[i];
            ch = 'a' + i;
        }
    }
 
    return ch;
}
 
string rearrangeString(string S)
{
    int N = S.size();
    if (N == 0)
        return "";
 
    vector<int> count(26, 0);
    for (auto& ch : S)
        count[ch - 'a']++;
 
    char ch_max = getMaxCountChar(count);
    int maxCount = count[ch_max - 'a'];
 
    // check if the result is possible or not
    if (maxCount > (n + 1) / 2)
        return "";
 
    string res(n, ' ');
    int ind = 0;
 
    // filling the most frequently occurring char in the
    // even indices
    while (maxCount) {
        res[ind] = ch_max;
        ind = ind + 2;
        maxCount--;
    }
 
    count[ch_max - 'a'] = 0;
 
    // now filling the other Chars, first
    // filling the even positions and then
    // the odd positions
    for (int i = 0; i < 26; i++) {
 
        while (count[i] > 0) {
 
            ind = (ind >= n) ? 1 : ind;
            res[ind] = 'a' + i;
            ind += 2;
            count[i]--;
        }
    }
 
    return res;
}

# Python program for rearranging characters in a string such
# that no two adjacent are same
 
# Function to find the char with maximum frequency in the given
# string
 
 
def getMaxCountChar(count):
    maxCount = 0
    for i in range(26):
        if count[i] > maxCount:
            maxCount = count[i]
            maxChar = chr(i + ord('a'))
 
    return maxCount, maxChar
 
# Main function for rearranging the characters
 
 
def rearrangeString(S):
    N = len(S)
 
    # if length of string is None return False
    if not N:
        return False
 
    # create a hashmap for the alphabets
    count = [0] * 26
    for char in S:
        count[ord(char) - ord('a')] += 1
 
    maxCount, maxChar = getMaxCountChar(count)
 
    # if the char with maximum frequency is more than the half of the
    # total length of the string than return False
    if maxCount > (N + 1) // 2:
        return False
 
    # create a list for storing the result
    res = [None] * N
 
    ind = 0
 
    # place all occurrences of the char with maximum frequency in
    # even positions
    while maxCount:
        res[ind] = maxChar
        ind += 2
        maxCount -= 1
 
    # replace the count of the char with maximum frequency to zero
    # as all the maxChar are already placed in the result
    count[ord(maxChar) - ord('a')] = 0
 
    # place all other char in the result starting from remaining even
    # positions and then place in the odd positions
    for i in range(26):
        while count[i] > 0:
            if ind >= N:
                ind = 1
            res[ind] = chr(i + ord('a'))
            ind += 2
            count[i] -= 1
 
    # convert the result list to string and return
    return ''.join(res)
 
 
# Driver Code
if __name__ == '__main__':
    str = 'bbbaa'
 
    # Function call
    res = rearrangeString(str)
    if res:
        print(res)
    else:
        print('Not possible')