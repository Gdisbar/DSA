556. Next Greater Element III
===================================
// Given a positive integer n, find the smallest integer which has exactly the 
// same digits existing in the integer n and is greater in value than n. If no 
// such positive integer exists, return -1.

// Note that the returned integer should fit in 32-bit integer, if there is a 
// valid answer but it does not fit in 32-bit integer, return -1.

// If you''re simply converting number to string then swapping the 1st lesser 
// element with it''s immediate greater to right it will fail for 

// Input : 230241
// Output : 230421
// Expected : 230412

//Algorithm

// 1)Traverse from right, if the input number is “534976”, we stop at 4(2) because 
// 	4 is smaller than next digit 9(3). If we do not find such a digit, then output 
// 	is “Not Possible”.,i=3
// 2)Now search the right side of above found digit ‘d’ for the smallest digit 
// 	greater than ‘d’. For “534976″, the right side of 4(2) contains “976”. 
// 	The smallest digit greater than 4 is 6(5). & swap them to get 536974 
// 3)Now sort all digits from position next to ‘d’ i.e i-1 to the end of number. 
// 	For above example, we sort digits in bold 536|974|. We get “536479” which is 
// 	the next greater number for input 534976.

//TC : nlogn

int nextGreaterElement(int n) {
        string s = to_string(n);
        int i=0;
        //1. find backwards “534976” we stop at 4(2) as 9(3)>4(2)
        for(i = s.size()-1;i>0;i--)
            if(s[i]>s[i-1])break;
        if(i==0)return -1;
        //2. first -> (4,at pos=2), second -> (9,pos=3)
        int ff = i-1,ss=i;
        //3. find the next greater than first, backward
        for(i=s.size()-1;i>ff;i--)
            if(s[i]>s[ff]){
                swap(s[i],s[ff]); //534976 ---> 536974 
                break;
            }
        //4. reverse after second i.e for 536|974|--> 536479
        reverse(s.begin()+ss,s.end());
        //5. Transform back
        if(stol(s)<=INT_MAX)return stol(s);
        else return -1;
    }