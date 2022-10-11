NAJPF - Pattern Find
===============================
Your task is so simple given a string and a pattern. You find the pattern on 
the given string. If found print how many time found the pattern and their index. 
Otherwise print ‘Not Found’

Input:          

The input   line consists of a number T (1 ≤ T ≤ 50) test cases.

For each case given two string number  A,B. the string and the pattern  
1 ≤|A|, |B| ≤10^6

All character will be lower case Latin character.  And |  | is the length of string.

Output:

For each case print the number  (found pattern from the given string) next line 
there position And Otherwise print 'Not Found' without quota.
There will a blank line between two cases.

Sample:

Input
    
3
ababab ab
aaaaa bbb
aafafaasf aaf
    
Output

3
1 3 5

Not Found

1
1


//Brute-force

int brute_force(string text,string pattern) {
  // let n be the size of the text and m the size of the
  // pattern

  for (i = 0; i < n; i++) {
    for (j = 0; j < m && i + j < n; j++)
      if (text[i + j] != pattern[j]) -1;
    // mismatch found, break the inner loop
    if (j == m) // match found
        return j;
  }
}

//Rabin-Karp 

#define HASH_BASE 0

vi rabinKarp(string &pattern, string &text, int inputBase) {
    int patternLen = pattern.size();
    int textLen = text.size();
    int i, j; //predefined iterators
    int patternHash = 0;
    int textHash = 0;
    int patternLenOut = 1;
    vi v;

    // hash of pattern len
    for(i=0;i<patternLen-1;++i) 
        patternLenOut=(patternLenOut*HASH_BASE)%inputBase;

    // Calculate hash value for pattern and text
    for(i=0;i<patternLen;++i) {
        patternHash=(HASH_BASE*patternHash+pattern[i])%inputBase;
        textHash=(HASH_BASE*textHash+text[i])%inputBase;
    }

    // Find the match
    for(i=0;i<=textLen-patternLen;++i) {
        if (patternHash==textHash) {
            for(j=0;j<patternLen;++j) {
                if (text[i+j]!=pattern[j])
                    break;
            }
            if (j==patternLen) v.pb(i+1);
                //cout << "Pattern is found at position: " << i + 1 << endl;
        }

        if (i<textLen-patternLen) {
            textHash=(HASH_BASE*(textHash-text[i]*patternLenOut)+text[i+patternLen])%inputBase;
            if (textHash<0)
                textHash=(textHash+inputBase);
        }
    }
    return v;
    
}

void solve() {
    string text,pattern;cin>>text>>pattern;
    vi v= rabinKarp(pattern,text,13);
    if(v.size()==0) cout<<"Not Found";
      else{
        cout<<v.size()<<endl;
        vout(v);
      }
      
}