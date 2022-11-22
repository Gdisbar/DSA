// left smaller --> TC:n*n --> TC: n, SC : n
while(!s.empty()&&s.top()>=a[i]) s.pop() //remove all larger element
if(s.empty()) LS[i]=s.top() //left smaller element of current
else LS[i]=0 // left smaller doesn't exist
s.push(a[i])
// right smaller -->  TC:n*n --> TC: n, SC : n
reverse(a) then find leftsmaller i.e RS


Find maximum difference between nearest left and right smaller elements
================================================================================
// Given an array of integers, the task is to find the maximum absolute difference 
// between the nearest left and the right smaller element of every element in the array. 

// Note: If there is no smaller element on right side or left side of any element 
// then we take zero as the smaller element. For example for the leftmost element, 
// the nearest smaller element on the left side is considered as 0. Similarly, for 
// rightmost elements, the smaller element on the right side is considered as 0.

// Examples: 

// Input : arr[] = {2, 1, 8}
// Output : 1
// Left smaller  LS[] {0, 0, 1}
// Right smaller RS[] {1, 0, 0}
// Maximum Diff of abs(LS[i] - RS[i]) = 1 

// Input  : arr[] = {2, 4, 8, 7, 7, 9, 3}
// Output : 4
// Left smaller   LS[] = {0, 2, 4, 4, 4, 7, 2}
// Right smaller  RS[] = {0, 3, 7, 3, 3, 3, 0}
// Maximum Diff of abs(LS[i] - RS[i]) = 7 - 3 = 4 

// Input : arr[] = {5, 1, 9, 2, 5, 1, 7}
// Output : 1

//TC : n , sc : n

//we push a[i] into stack & pop stack until we find some smaller element previously 
//inserted into stack which is lesser than current element,if we get such element
//we've found out the smaller element

void leftSmaller(int arr[], int n, int SE[])
{
    // Create an empty stack
    stack<int>S;
 
    // Traverse all array elements
    // compute nearest smaller elements of every element
    for (int i=0; i<n; i++)
    {
        // Keep removing top element from S while the top
        // element is greater than or equal to arr[i]
        while (!S.empty() && S.top() >= arr[i])
            S.pop();
 
        // Store the smaller element of current element
        if (!S.empty())
            SE[i] = S.top();
 
        // If all elements in S were greater than arr[i]
        else
            SE[i] = 0;
 
        // Push this element
        S.push(arr[i]);
    }
}
 
// Function returns maximum difference b/w Left &
// right smaller element
int findMaxDiff(int arr[], int n)
{
    int LS[n]; // To store left smaller elements
 
    // find left smaller element of every element
    leftSmaller(arr, n, LS);
 
    // find right smaller element of every element
    // first reverse the array and do the same process
    int RRS[n]; // To store right smaller elements in
                // reverse array
    reverse(arr, arr + n);
    leftSmaller(arr, n, RRS);
 
    // find maximum absolute difference b/w LS & RRS
    // In the reversed array right smaller for arr[i] is
    // stored at RRS[n-i-1]
    int result = -1;
    for (int i=0 ; i< n ; i++)
        result = max(result, abs(LS[i] - RRS[n-1-i]));
 
    // return maximum difference b/w LS & RRS
    return result;
}