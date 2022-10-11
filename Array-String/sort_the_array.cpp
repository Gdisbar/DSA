B. Sort the Array | Codeforces Round #258 (Div. 2)
======================================================
// is it possible to sort the array a (in increasing order) by reversing exactly one 
// segment of a? See definitions of segment and reversing in the notes.

// Input

// The first line of the input contains an integer n (1≤n≤105) — the size of array a.

// The second line contains n distinct space-separated integers: a[1],a[2],...,a[n] 
// (1≤a[i]≤109).

// Output

// Print "yes" or "no" (without quotes), depending on the answer.

// If your answer is "yes", then also print two space-separated integers denoting start 
// and end (start must not be greater than end) indices of the segment to be reversed. 
// If there are multiple ways of selecting these indices, print any of them.

// Examples

// 3
// 3 2 1
// yes
// 1 3
mp [ {1,0} {2,1} {3,2} ]
a [ 2 1 0 ]
a [ 0 1 2 ]

// 4
// 2 1 3 4
// yes
// 1 2
mp [ {1,0} {2,1} {3,2} {4,3} ]
a [ 1 0 2 3 ]
a [ 0 1 2 3 ]

// 4
// 3 1 2 4
// no
mp [ {1,0} {2,1} {3,2} {4,3} ]
a [ 2 0 1 3 ]
a [ 1 0 2 3 ]

// 2
// 1 2
// yes
// 1 1


// 6
// 1 2 4 3 5 6
// yes
// 3 4
mp [ {1,0} {2,1} {3,2} {4,3} {5,4} {6,5} ]
a [ 0 1 3 2 4 5 ]
a [ 0 1 2 3 4 5 ] //reverse(a.begin()+s,a.begin()+e+1)



// if from a given sorted array, if reverse a segment, then the remaining array will 
// be arranged in following way. First increasing sequence, then decreasing, then 
// again increasing.

//here we're mapping -> a[i]->i for those which are relatively sorted , as we can't
//check below 2 conditions simultaneously , we use this map approach
// a[i]<a[i+1]&&a[i]<a[i-1] , s=a[0],e=a[0] ---> 2 1 3 4
// a[i]>a[i-1]&&a[i]>a[i+1] ,s=-1,e=-1 ---> 1 2 4 3 5 6

void solve(){
  int n;cin>>n;
  vi a(n),b(n);
  rep(i,0,n-1){
    cin>>a[i];
    b[i]=a[i];
  }
  sort(all(b));
  map<int,int> mp;
  rep(i,0,n-1){
    mp[b[i]]=i;
  }
  rep(i,0,n-1){
    a[i]=mp[a[i]];
  }
  int s=-1,e=-1;
  rep(i,0,n-1){
    if(a[i]!=i){
        s=i;
        break;
    }
  }
  repi(i,n-1,0){
    if(a[i]!=i){
        e=i;
        break;
    }
  }
  if(s==-1||e==-1){
    cout<<"yes"<<endl;
    cout<<1<<" "<<1;
  }else{
    reverse(a.begin()+s,a.begin()+e+1);
    bool f=true;
    rep(i,0,n-1){
        if(a[i]!=i){
            f=false;
        }
    }
    if(f){
        cout<<"yes"<<endl;
        cout<<s+1<<" "<<e+1<<endl;
    }else{
        cout<<"no";
    }
  }
        
}