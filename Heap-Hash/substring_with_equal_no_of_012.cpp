Count Substrings with equal number of 0s, 1s and 2s
==========================================================
Given a string which consists of only 0s, 1s or 2s, count the number of 
substrings that have equal number of 0s, 1s and 2s.

Examples: 

Input  :  str = “0102010”
Output :  2
Explanation : Substring str[2, 4] = “102” and 
              substring str[4, 6] = “201” has 
              equal number of 0, 1 and 2

Input : str = "102100211"
Output : 5

Let zc[i] denotes number of zeros between index 1 and i
    oc[i] denotes number of ones between index 1 and i
    tc[i] denotes number of twos between index 1 and i
for substring str[i, j] to be counted in result we should have :
    zc[i] – zc[j-1] = oc[i] – oc[j-1] = tc[i] - tc[j-1]
we can write above relation as follows :
z[i] – o[i] = z[j-1] – o[j-1]    and
z[i] – t[i] = z[j-1] – t[j-1]


int getSubstringWithEqual012(string str)
{
    int n = str.length();
 
    // map to store, how many times a difference pair has occurred previously
    map< pair<int, int>, int > mp;
    mp[make_pair(0, 0)] = 1;
    int zc = 0, oc = 0, tc = 0;
    int res = 0; 
    for (int i = 0; i < n; ++i){
        if (str[i] == '0') zc++;
        else if (str[i] == '1') oc++;
        else tc++; 
        // making pair of differences (z[i] - o[i],z[i] - t[i])
        pair<int, int> tmp = make_pair(zc - oc,zc - tc);
        // Count of previous occurrences of above pair indicates that the 
        // subarrays forming from every previous occurrence to this occurrence 
        // is a subarray with equal number of 0's, 1's and 2's
        res = res + mp[tmp];
        // increasing the count of current difference pair by 1
        mp[tmp]++;
    }
 
    return res;
}


Maximum consecutive one’s (or zeros) in a binary array
==========================================================
Given binary array, find count of maximum number of consecutive 1’s present 
in the array.

Examples : 

Input  : arr[] = {1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1}
Output : 4

Input  : arr[] = {0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
Output : 1

int getMaxLength(bool arr[], int n)
{
    int count = 0; //initialize count
    int result = 0; //initialize max
 
    for (int i = 0; i < n; i++)
    {
        // Reset count when 0 is found
        if (arr[i] == 0)
            count = 0;
 
        // If 1 is found, increment count
        // and update result if count becomes
        // more.
        else
        {
            count++;//increase count
            result = max(result, count);
        }
    }
 
    return result;
}
