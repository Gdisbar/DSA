No Prefix Set
// ===============
// There is a given list of strings where each string contains only lowercase 
// letters from a-j, inclusive. The set of strings is said to be a 
// GOOD SET if no string is a prefix of another string. In this case, 
// print GOOD SET. Otherwise, print BAD SET on the first line followed 
// by the string being checked.

// Note If two strings are identical, they are prefixes of each other.

// Example
// words=["abcd","bcd","abcde","bcde"]

// Here 'abcd' is a prefix of 'abcde' and 'bcd' is a prefix 
// of 'bcde'. Since 'abcde' is tested first, print 

// BAD SET  
// abcde

// words=["ab","bc","cd"]

// No string is a prefix of another so print

// GOOD SET 

// Sample Input00

// STDIN     Function
// -----     --------
// 7         words[] size n = 7
// aab       words = ['aab', 'defgab', 'abcde', 'aabcde', 'bbbbbbbbbb', 'jabjjjad']
// defgab  
// abcde
// aabcde
// cedaaa
// bbbbbbbbbb
// jabjjjad

// Sample Output00

// BAD SET
// aabcde

// Explanation
// 'aab' is prefix of 'aabcde' so it is a BAD SET and fails at string 'aabcde'.

// Sample Input01

// 4
// aab
// aac
// aacghgh
// aabghgh

// Sample Output01

// BAD SET
// aacghgh

// Explanation
// 'aab' is a prefix of 'aabghgh', and aac' is prefix of 'aacghgh'. 
// The set is a BAD SET. 'aacghgh' is tested before 'aabghgh', so and 
// it fails at 'aacghgh'.


struct node{
     node * children[26] = {NULL};
     int cnt = 0;
     bool end = false;
};

struct tracker{
    bool had_endings = false;
    int cnt = 0;
};

void addString(node * root, string name, int cur,tracker * tr)
{
   
    root->cnt += 1;
    if(cur == name.size())
    {
        tr->cnt = root->cnt;
        root->end = true;
        return;
    }
    if(root->end){
        tr->had_endings = true;
    }
    if(!root->children[name[cur] - 'a']) root->children[name[cur] - 'a'] = new node();
    addString(root->children[name[cur] - 'a'],name, cur + 1,tr);
}

void noPrefix(vector<string> words) {
    node * root = new node();
    
    for(string s: words){
        tracker track;
        addString(root, s, 0,&track);
        if(track.had_endings || track.cnt > 1){
            cout << "BAD SET" << "\n";
            cout << s;
            return;
        }
        
    }


    cout << "GOOD SET";
}