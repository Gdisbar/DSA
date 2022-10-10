Characteristics of counting sort:
========================================
// 1)Counting sort makes assumptions about the data, for example, it assumes that 
// values are going to be in the range of 0 to 10 or 10 – 99 etc, 
// 2)input data will be all real numbers.
// 3)this sorting algorithm is not a comparison-based algorithm, 
// it hashes the value in a temporary count array and uses them for sorting.
// It uses a temporary array making it a non In Place algorithm.

//TC : n + range , SC : n , works for -ve & duplicates

void countSort(vector<int>& arr){

    int max = *max_element(arr.begin(), arr.end());
    int min = *min_element(arr.begin(), arr.end());
    int range = max - min + 1; // this sorting works on predefined range
 
    vector<int> count(range), output(arr.size()); // we also need predefined memory size
    for (int i = 0; i < arr.size(); i++) // frequency count,but with some hashing
        count[arr[i] - min]++;
 
    for (int i = 1; i < count.size(); i++) //if we have count[x]=1,cout[x+1]=1,count[x+2]=1 --> count[x+2]=3
        count[i] += count[i - 1];      // while printing from right to left,we can place them together
 
    for (int i = arr.size() - 1; i >= 0; i--) { 
        output[count[arr[i] - min] - 1] = arr[i];
        count[arr[i] - min]--; // if we have count[1]=3 then we write 1 1 1 i.e 3 times 1
    }
 
    for (int i = 0; i < arr.size(); i++)
        arr[i] = output[i];
}