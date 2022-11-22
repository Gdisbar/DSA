print all permutations of a given string
===========================================

// Function to print permutations of string 
// This function takes three parameters: 
// 1. String 
// 2. Starting index of the string 
// 3. Ending index of the string. 
void permute(string a, int l, int r) 
{ 
    // Base case 
    if (l == r) 
        cout<<a<<endl; 
    else
    { 
        // Permutations made 
        for (int i = l; i <= r; i++) 
        { 
  
            // Swapping done 
            swap(a[l], a[i]); 
  
            // Recursion called 
            permute(a, l+1, r); 
  
            //backtrack 
            swap(a[l], a[i]); 
        } 
    } 
} 
  
// Driver Code 
int main() 
{ 
    string str = "ABC"; 
    int n = str.size(); 
    permute(str, 0, n-1); 
    return 0; 
} 

// Output

// ABC
// ACB
// BAC
// BCA
// CBA
// CAB

// Algorithm Paradigm: Backtracking 

// Time Complexity: O(n*n!) Note that there are n! permutations and it requires 
// O(n) time to print a permutation.

// Auxiliary Space: O(r – l)

// # Python program to print all permutations with 
// # duplicates allowed 
  
def toString(List): 
    return ''.join(List) 
  

def permute(a, l, r): 
    if l==r: 
        print (toString(a))
    else: 
        for i in range(l,r): 
            a[l], a[i] = a[i], a[l] 
            permute(a, l+1, r) 
            a[l], a[i] = a[i], a[l] 
  
//# Driver program to test the above function 
string = "ABC"
n = len(string) 
a = list(string) 
permute(a, 0, n) 

47. Permutations II
===================================================================
// Given a collection of numbers, nums, that might contain duplicates, 
// return all possible unique permutations in any order.

 

// Example 1:

// Input: nums = [1,1,2]
// Output:
// [[1,1,2],
//  [1,2,1],
//  [2,1,1]]

// Example 2:

// Input: nums = [1,2,3]
// Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]


void recursion(vector<int> num, int i, int n, vector<vector<int> > &res) {
        if (i == n-1) {
            res.push_back(num);
            return;
        }
        for (int k = i; k < n; k++) {
            if (i != k && num[i] == num[k]) continue;
            swap(num[i], num[k]);
            recursion(num, i+1, n, res);
        }
    }
    vector<vector<int> > permuteUnique(vector<int> &num) {
        sort(num.begin(), num.end());
        vector<vector<int> >res;
        recursion(num, 0, num.size(), res);
        return res;
    }


void solve(){
    string s = "ACBC";
    vector<char> v(s.begin(), s.end()); //convert int to chan
    vector<vector<char> >res = permuteUnique(v);
    for(int i =0;i<res.size();++i){
        string s(res[i].begin(),res[i].end());
        cout<<s<<endl;
    }
}

// TC : n*n!
// SC : n

// Output

// ABCC
// ACBC
// ACCB
// BACC
// BCAC
// BCCA
// CABC
// CACB
// CBAC
// CBCA
// CCAB
// CCBA