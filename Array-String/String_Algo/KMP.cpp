KMP
=====
ABCDABC
prefix : A,AB,ABC,ABCD, ... more
suffix : C,BC,ABC,DABC, ... more

we''ve a match --> ABC

lps[i] = the longest proper prefix of pat[0...i]  which is also a suffix of pat[0...i]


       A  A  A  A
lps = [0, 1, 2, 3]

      A  B  C  D  E 
lps= [0, 0, 0, 0, 0]

i=    0  1  2  3  4  5  6  7  8  9  10
s=    A  A  B  A  A  C  A  A  B  A  A 
lps= [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]

i=1 --> match(i=0) --> lps[1]=length upto (i=0) = 1
i=3 --> match(i=0) --> lps[3]=length upto (i=0) = 1
i=4 --> match(i=0-1) --> lps[4]=length upto (i=0-1) = 2
i=6 --> match(i=0)   --> lps[6]=length upto (i=0) = 1
i=7 --> math(i=0-1)  --> lps[7]=length upto (i=0-1) = 2
i=8 --> match(i=0-2) --> lps[8]=length upto (i=0-2) = 3
i=9 --> match(i=0-3) --> lps[9]=length upto (i=0-3) = 4
i=10 --> match(i=0-4) --> lps[10]=length upto (i=0-4) = 5


      A  A  A  C  A  A  A  A  A  C 
lps= [0, 1, 2, 0, 1, 2, 3, 3, 3, 4] --> this one is an example of duplicate match

      A  A  A  B  A  A  A 
lps= [0, 1, 2, 0, 1, 2, 3]

void lps_func(string pattern, vector<int>&lps){
    lps[0] = 0;     
    // length of the previous longest prefix suffix              
    int j = 0; //matched length , we don't need to search this part again
    int i=1;
    while (i<pattern.length()){
        if(pattern[i]==pattern[j]){   
            j++;
            lps[i] = j;
            i++;
         //   continue;
        }
        else{                   
            if(j==0){         
                lps[i] = 0;
                i++;
            //    continue;
            }
            else{  
            //AAACAAAA and i = 7 , we start matching from i=6            
                j = lps[j-1];
            //    continue;
            }
        }
    }
}

void kmp(string pattern,string text,vector<int> &v){
    int n = text.length();
    int m = pattern.length();
    vector<int>lps(m);
    
    lps_func(pattern,lps); // This function constructs the lps array.
    
    int i=0,j=0;
    while(i<n){
        if(pattern[j]==text[i]){i++;j++;} // If there is a match continue.
        if (j == m) { // found pattern
            v.pb(i-m+1);   // and update j as lps of last matched character.
            j = lps[j - 1]; //already matched upto lps[j-1]
        } 
        // If there is a mismatch ,after j matches
        else if (i < n && pattern[j] != text[i]) {  
            if (j == 0)  // if j becomes 0 then simply increment the index i
                i++;
            else  //Do not match lps[0..lps[j-1]] characters,they will match anyway
                j = lps[j - 1];  //Update j as lps of last matched character
        }
    }
    
}