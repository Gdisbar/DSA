Largest number in K swaps
===========================
// Given a positive integer, find the maximum integer possible by doing at-most 
// K swap operations on its digits.


// Examples: 

// Input: M = 254, K = 1
// Output: 524
// Swap 5 with 2 so number becomes 524

// Input: M = 254, K = 2
// Output: 542
// Swap 5 with 2 so number becomes 524
// Swap 4 with 2 so number becomes 542

// Input: M = 68543, K = 1 
// Output: 86543
// Swap 8 with 6 so number becomes 86543

// Input: M = 7599, K = 2
// Output: 9975
// Swap 9 with 5 so number becomes 7995
// Swap 9 with 7 so number becomes 9975

// Input: M = 76543, K = 1 
// Output: 76543
// Explanation: No swap is required.

// Input: M = 129814999, K = 4
// Output: 999984211
// Swap 9 with 1 so number becomes 929814991
// Swap 9 with 2 so number becomes 999814291
// Swap 9 with 8 so number becomes 999914281
// Swap 1 with 8 so number becomes 999984211


def swap(string, i, j):
    return (string[:i] + string[j] +string[i + 1:j] +string[i] + string[j + 1:])
 
// # function to find maximum integer possible by doing at-most K swap
// # operations on its digits
def findMaximumNum(string, k, maxm):
    if k == 0:
        return
    n = len(string)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if string[i] < string[j]:
                string = swap(string, i, j)
                if string > maxm[0]:
                    maxm[0] = string
                //# recurse of the other k - 1 swaps
                findMaximumNum(string, k - 1, maxm)
                //# backtrack
                string = swap(string, i, j)
 

if __name__ == "__main__":
    string = "129814999"
    k = 4
    maxm = [string]
    findMaximumNum(string, k, maxm)
    print(maxm[0])
 

// Output: 

// 999984211

 

// Complexity Analysis:  

//     Time Complexity: O((n^2)^k). 
//     For every recursive call n^2 recursive calls is generated until the value 
//     of k is 0. So total recursive calls are O((n^2)^k).
//     Space Complexity:O(n). 
//     This is the space required to store the output string.



void findMaximumNum(string str, int k,string& max, int i){
    if (k == 0)
        return;
    int n = str.length();
    char maxm = str[i];
    for (int j = i + 1; j < n; j++) {
           if(maxm<str[j])
            maxm=str[j];
    }
    if (maxm != str[i])
        --k;
 
    // search this maximum among the rest from behind
    //first swap the last maximum digit if it occurs more than 1 time
   //example str= 1293498 and k=1 then max string is 9293418 instead of 9213498
    for (int j = n-1; j >=i; j--) {
        if (str[j] == maxm) {
            swap(str[i], str[j]);
            // If current num is more than  maximum so far
            if (str.compare(max) > 0)
                max = str;
            findMaximumNum(str, k, max, i + 1);
            swap(str[i], str[j]);
        }
    }
}
 
// Driver code
int main()
{
    string str = "129814999";
    int k = 4;
    string max = str;
    findMaximumNum(str, k, max, 0);
    cout << max << endl;
    return 0;
}
// Output: 

// 999984211


// Complexity Analysis:  

//     Time Complexity: O(n^k). 
//     For every recursive call n recursive calls is generated until the value 
//     of k is 0. So total recursive calls are O((n)^k).
//     Space Complexity: O(n). 
//     The space required to store the output string.
