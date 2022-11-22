Print all subsequences of a string
========================================
// Here, we discuss much easier and simpler iterative approach which is similar 
// to Power Set. We use bit pattern from binary representation of 1 
// to 2^length(s) – 1.

// input = “abc” 
// Binary representation to consider 1 to (2^3-1), i.e 1 to 7. 
// Start from left (MSB) to right (LSB) of binary representation and append 
// characters from input string which corresponds to bit value 1 in binary 
// representation to Final subsequence string sub.

// Example: 

// 001 => abc . Only c corresponds to bit 1. So, subsequence = c. 
// 101 => abc . a and c corresponds to bit 1. So, subsequence = ac.
// binary_representation (1) = 001 => c 
// binary_representation (2) = 010 => b 
// binary_representation (3) = 011 => bc 
// binary_representation (4) = 100 => a 
// binary_representation (5) = 101 => ac 
// binary_representation (6) = 110 => ab 
// binary_representation (7) = 111 => abc

 // function to find subsequence
string subsequence(string s, int binary, int len)
{
    string sub = "";
    for (int j = 0; j < len; j++)
 
        // check if jth bit in binary is 1
        if (binary & (1 << j))
 
            // if jth bit is 1, include it
            // in subsequence
            sub += s[j];
 
    return sub;
}
 
// function to print all subsequences
void possibleSubsequences(string s){
 
    // map to store subsequence
    // lexicographically by length
    map<int, set<string> > sorted_subsequence;
 
    int len = s.size();
     
    // Total number of non-empty subsequence
    // in string is 2^len-1
    int limit = pow(2, len);
     
    // i=0, corresponds to empty subsequence
    for (int i = 1; i <= limit - 1; i++) {
         
        // subsequence for binary pattern i
        string sub = subsequence(s, i, len);
         
        // storing sub in map
        sorted_subsequence[sub.length()].insert(sub);
    }
 
    for (auto it : sorted_subsequence) {
         
        // it.first is length of Subsequence
        // it.second is set<string>
        cout << "Subsequences of length = "
             << it.first << " are:" << endl;
              
        for (auto ii : it.second)
             
            // ii is iterator of type set<string>
            cout << ii << " ";
         
        cout << endl;
    }
}
// Time Complexity : O(2^{n} * l), where n is length of string to find 
// subsequences and l is length of binary string.
// Space Complexity: O(1)

//Approach - 2

//  Approach is to get the position of rightmost set bit and reset that bit 
//  after appending corresponding character from given string to the subsequence 
//  and will repeat the same thing till corresponding binary pattern has no set bits.

// If input is s = “abc” 
// Binary representation to consider 1 to (2^3-1), i.e 1 to 7. 
// 001 => abc . Only c corresponds to bit 1. So, subsequence = c 
// 101 => abc . a and c corresponds to bit 1. So, subsequence = ac.
// Let us use Binary representation of 5, i.e 101. 
// Rightmost bit is at position 1, append character at beginning of sub = c ,
// reset position 1 => 100 
// Rightmost bit is at position 3, append character at beginning of sub = ac ,
// reset position 3 => 000 
// As now we have no set bit left, we stop computing subsequence.

// Example:

// binary_representation (1) = 001 => c 
// binary_representation (2) = 010 => b 
// binary_representation (3) = 011 => bc 
// binary_representation (4) = 100 => a 
// binary_representation (5) = 101 => ac 
// binary_representation (6) = 110 => ab 
// binary_representation (7) = 111 => abc

// function to find subsequence
string subsequence(string s, int binary)
{
    string sub = "";
    int pos;
     
    // loop while binary is greater than 0
    while(binary>0)
    {
        // get the position of rightmost set bit
        pos=log2(binary&-binary)+1;
         
        // append at beginning as we are
        // going from LSB to MSB
        sub=s[pos-1]+sub;
         
        // resets bit at pos in binary
        binary= (binary & ~(1 << (pos-1)));
    }
    reverse(sub.begin(),sub.end());
    return sub;
}
 
// function to print all subsequences
void possibleSubsequences(string s){
 
    // map to store subsequence
    // lexicographically by length
    map<int, set<string> > sorted_subsequence;
 
    int len = s.size();
     
    // Total number of non-empty subsequence
    // in string is 2^len-1
    int limit = pow(2, len);
     
    // i=0, corresponds to empty subsequence
    for (int i = 1; i <= limit - 1; i++) {
         
        // subsequence for binary pattern i
        string sub = subsequence(s, i);
         
        // storing sub in map
        sorted_subsequence[sub.length()].insert(sub);
    }
 
    for (auto it : sorted_subsequence) {
         
        // it.first is length of Subsequence
        // it.second is set<string>
        cout << "Subsequences of length = "
            << it.first << " are:" << endl;
             
        for (auto ii : it.second)
             
            // ii is iterator of type set<string>
            cout << ii << " ";
         
        cout << endl;
    }
}
 
// driver function
int main()
{
    string s = "aabc";
    possibleSubsequences(s);
     
    return 0;
}

Subsequences of length = 1 are:
a b c 
Subsequences of length = 2 are:
aa ab ac bc 
Subsequences of length = 3 are:
aab aac abc 
Subsequences of length = 4 are:
aabc 

// Time Complexity: O(2^{n} * b) , where n is the length of string to find 
// subsequence and b is the number of set bits in binary string.
// Auxiliary Space: O(n)