43. Multiply Strings
=======================
// Given two non-negative integers num1 and num2 represented as strings, return the 
// product of num1 and num2, also represented as a string.

// Note: You must not use any built-in BigInteger library or convert the inputs to 
// integer directly.

 

// Example 1:

// Input: num1 = "2", num2 = "3"
// Output: "6"

// Example 2:

// Input: num1 = "123", num2 = "456"
// Output: "56088"

string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") return "0";
        
        vector<int> res(num1.size()+num2.size(), 0);
        
        for (int i = num1.size()-1; i >= 0; i--) {
            for (int j = num2.size()-1; j >= 0; j--) {
                res[i + j + 1] += (num1[i]-'0') * (num2[j]-'0');
                res[i + j] += res[i + j + 1] / 10;
                res[i + j + 1] %= 10;
            }
        }
        // removing "0" at beginning
        int i = 0;
        string ans = "";
        while (res[i] == 0) i++;
        while (i < res.size()) ans += to_string(res[i++]);
        
        return ans;
    }

def multiply(self, num1: str, num2: str) -> str:
        res=[0]*(len(num1)+len(num2))
        for i in range(len(num1)-1,-1,-1):
            for j in range(len(num2)-1,-1,-1):
                res[i+j+1]+=(ord(num1[i])-ord("0"))*(ord(num2[j])-ord("0"))
                res[i+j]+=res[i+j+1]//10
                res[i+j+1]%=10
        res = "".join(map(str, res))
        return "0" if not res.lstrip("0") else res.lstrip("0")