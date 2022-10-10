Check whether Arithmetic Progression can be formed from the given array
==========================================================================
// Given an array of n integers. The task is to check whether an arithmetic 
// progression can be formed using all the given elements. If possible print 
// “Yes”, else print “No”.

// Examples: 

// Input : arr[] = {0, 12, 4, 8}
// Output : Yes
// Rearrange given array as {0, 4, 8, 12} 
// which forms an arithmetic progression.

// Input : arr[] = {12, 40, 11, 20}
// Output : No


bool checkIsAP(int arr[], int n)
{
    unordered_set<int> st;
    int maxi = INT_MIN;
    int mini = INT_MAX;
    for (int i=0;i<n;i++) {
        maxi = max(arr[i], maxi);
        mini = min(arr[i], mini);
        st.insert(arr[i]);
    }
    // FINDING THE COMMON DIFFERENCE
    int diff = (maxi - mini) / (n - 1);
    int count = 0;
 
    // CHECK IF PRESENT IN THE HASHSET OR NOT , second max = max - diff .. so on
    while (st.find(maxi)!=st.end()) {
        count++;
        maxi = maxi - diff;
    }
    if (count == n)
        return true;
 
    return false;
}
 