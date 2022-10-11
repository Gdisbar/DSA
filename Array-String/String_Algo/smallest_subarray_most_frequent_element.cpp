Smallest subarray with all occurrences of a most frequent element
====================================================================
Given an array, A. Let x be an element in the array. x has the maximum 
frequency in the array. Find the smallest subsegment of the array which also 
has x as the maximum frequency element.

Examples: 
 

Input :  arr[] = {4, 1, 1, 2, 2, 1, 3, 3} 
Output :   1, 1, 2, 2, 1
The most frequent element is 1. The smallest
subarray that has all occurrences of it is
1 1 2 2 1

Input :  A[] = {1, 2, 2, 3, 1}
Output : 2, 2
Note that there are two elements that appear
two times, 1 and 2. The smallest window for
1 is whole array and smallest window for 2 is
{2, 2}. Since window for 2 is smaller, this is
our output.

void smallestSubsegment(int a[], int n)
{
    // To store left most occurrence of elements
    unordered_map<int, int> left;
 
    // To store counts of elements
    unordered_map<int, int> count;
 
    // To store maximum frequency
    int mx = 0;
 
    // To store length and starting index of
    // smallest result window
    int mn, strindex;
 
    for (int i = 0; i < n; i++) {
 
        int x = a[i];
 
        // First occurrence of an element,
        // store the index
        if (count[x] == 0) {
            left[x] = i;
            count[x] = 1;
        }
 
        // increase the frequency of elements
        else
            count[x]++;
 
        // Find maximum repeated element and
        // store its last occurrence and first
        // occurrence
        if (count[x] > mx) {
            mx = count[x];
            mn = i - left[x] + 1; // length of subsegment
            strindex = left[x];
        }
 
        // select subsegment of smallest size
        else if (count[x] == mx && i - left[x] + 1 < mn) {
            mn = i - left[x] + 1;
            strindex = left[x];
        }
    }
 
    // Print the subsegment with all occurrences of
    // a most frequent element
    for (int i = strindex; i < strindex + mn; i++)
        cout << a[i] << " ";
}
 