Chef and Secret Password | Problem Code: SECPASS
===================================================
Chef knows that the secret password is a non-empty prefix of the string S
Also, he knows that:
if a prefix occurs in S more often as a substring, then the probability that 
this prefix is the secret password is higher (the probability that a chosen 
prefix is the secret password is an increasing function of its number of 
occurrences in S)if two prefixes have the same number of occurrences in S, 
then the longer prefix has a higher probability of being the secret password''

Chef wants to guess the secret password with the maximum possible probability of 
success. Help Chef and find the prefix for which this probability is maximum.

3
3
abc
5
thyth
5
abcbc

abc
th
abcbc

If our answer length is K : then we have 1st char of string followed by K-1 letters
& this prefix substring should appear maximum times . If we start from last it''s 
guranteed that ther is no occurrences of 1st char after current index

//This solution runs in O(∣S∣)

// Z-function = longest substr starting at k which is also prefix of the string
//0 1 0 0 1 2 0 1 2 3  --> lps of s
//0 1 0 0 2 1 0 3 1 0  --> z function of s
  0123456789
s=aabxaayaab
z=0100210310 
//# of character matched,those that are not mentioned here are single char mismatched
compare(s[0-1]='aa' & s[4-5]='aa') = z[4]=2 
compare(s[1-2]='ab' & s[5-6]='ay') = z[5]=1
compare(s[0-2]='aab' & s[7-9]='aab')=z[7]=3
compare(s[0-1]='aa' & s[8-9]='ab')  =z[8]=1

txt=x|abcabz|abc
pat=abc => 3
combined=  abc$x|abcabz|abc
       z=  00000|300000|300 ---> here we''ve 2 match (pat length same as z value)
                             index of match = z index - pat length

txt=xabcabxabc , pat=abc

upto i=5,l=i,r=i,z[i]=0 i.e no match found --> i=5,l=5,r=5,z[5]=0
i=6 ,z[6]=0 ---> now s[i+z[i]]==s[z[i]]
s[6]==s[0] --> z[6]= 1 --> 'x'
s[7]==s[1] --> z[6]= 2 --> 'xa'
s[8]==s[2] --> z[6]= 3 --> 'xab'
s[9]==s[3] --> z[6]= 4 --> 'xabc'
for rest upto i=7-9 --> l=6,r=6+4=10


void solve() {
    int n,z[100000];
    string s;
    cin>>n>>s;
    for (int i=1,l=0,r=0;i<n;++i){
      z[i]=max(0,min(z[i-l],r-i));
      while(i+z[i]<n&&s[i+z[i]]==s[z[i]])
        z[i]++;
      if(i+z[i]>r){
        l=i;
        r=i+z[i];
      }
    }
    int res=1000000000;
    z[0]=0;
    rep(i,0,n-1)
      if (z[i]!=0)
        res=min(res, z[i]);
    if(res==1000000000) cout << s;
    else rep(i,0,res-1) cout << s[i];
}

// KMP (return # of matching) + BS , almost 6x slower than z-function


void solve(){
	int n;
    string s;
	cin >> n >> s;
    int cnt=0;
    for(int i=0;i<n;i++) // initally occurrence of 1st character in the string
        if(s[i]==s[0])
            cnt++;

    int start=1,end=n,mid;

    while(start<end){
        mid=(start+end+1)/2; // makes 7-4, 8-4,3-2 ,1-1.. adding 1 dont forget

        int tmp=KMPSearch(s.substr(0,mid),s); //returns the number of matching points

        if(tmp==cnt) 
            start=mid;
        else
            end=mid-1;
    }
    cout << s.substr(0,start);
}