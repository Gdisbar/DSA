Sort an array of 0s, 1s and 2s
=================================
//Dutch National Flag Problem

void sort012(int a[], int n)
{
    int lo = 0;
    int hi = n - 1;
    int mid = 0;
 
    while (mid <= hi) {
        switch (a[mid]) {
 
        // If the element is 0
        case 0:
            swap(a[lo++], a[mid++]);
            break;
 
        // If the element is 1 .
        case 1:
            mid++;
            break;
 
        // If the element is 2
        case 2:
            swap(a[mid], a[hi--]);
            break;
        }
    }
}

//M-2
count number of zeros,ones & twos(n-zeros-ones)  now place them at 
beginning , mid & end 


905. Sort Array By Parity
============================
Given an integer array nums, move all the even integers at the beginning of 
the array followed by all the odd integers.

Return any array that satisfies this condition.

 

Example 1:

Input: nums = [3,1,2,4]
Output: [2,4,3,1]
Explanation: The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.

Example 2:

Input: nums = [0]
Output: [0]

vector<int> sortArrayByParity(vector<int>& a) {
        int l=0,h=a.size()-1;
        while(l<=h){
            if(a[l]%2==1&&a[h]%2==0){
                swap(a[l],a[h]);
                l++;
                h--;
            }
            if(a[l]%2==0) l++;
            if(a[h]%2==1) h--;
        }
        return a;
    }

//same but 4ms faster than previous one

vector<int> sortArrayByParity(vector<int>& a) {
        for (int i = 0, j = 0; j < a.size(); j++)
            if (a[j] % 2 == 0) swap(a[i++], a[j]);
        return a;
    }

A Program to check if strings are rotations of each other or not
===================================================================

str1="ABCD" , str2="CDAB" --> "Yes rotation of each other"
/* Function checks if passed strings (str1 and str2) are rotations of each other */
// TC : n*n , SC: n
bool areRotations(string str1, string str2)
{
   /* Check if sizes of two strings are same */
   if (str1.length() != str2.length())
        return false;
 
   string temp = str1 + str1; //ABCDCDAB
  return (temp.find(str2) != string::npos);
}

//Using queue --> Reduce TC to linear
// TC : n1 + n2 , SC : n
bool check_rotation(string s, string goal)
{
    if (s.size() != goal.size())
        return false;
    queue<char> q1,q2;
    for (int i = 0; i < s.size(); i++) {
        q1.push(s[i]);
    }
    queue<char> q2;
    for (int i = 0; i < goal.size(); i++) {
        q2.push(goal[i]);
    }
    int k = goal.size();
    while (k--) {
        char ch = q2.front(); //DAB --> ABC
        q2.pop();
        q2.push(ch);   //DABC --> ABCD
        if (q2 == q1)
            return true;
    }
    return false;
}