Painter''s Partition Problem
===============================

Given 2 integers A and B and an array of integars C of size N.

Element C[i] represents length of ith board.

You have to paint all N boards [C0, C1, C2, C3 … CN-1]. There are A 
painters available and each of them takes B units of time to paint 
1 unit of board.

Calculate and return minimum time required to paint all boards under the 
constraints that any painter will only paint contiguous sections of board.

        2 painters cannot share a board to paint. That is to say, a board
        cannot be painted partially by one painter, and partially by another.
        A painter will only paint contiguous boards. Which means a
        configuration where painter 1 paints board 1 and 3 but not 2 is
        invalid.

Return the ans % 10000003




Input Format

The first argument given is the integer A.
The second argument given is the integer B.
The third argument given is the integer array C.

Output Format

Return minimum time required to paint all boards under the constraints that any 
painter will only paint contiguous sections of board % 10000003.

Constraints

1 <=A <= 1000
1 <= B <= 10^6
1 <= C.size() <= 10^5
1 <= C[i] <= 10^6

For Example

Input 1:
    A = 2
    B = 5
    C = [1, 10]
Output 1:
    50
Explanation 1:
    Possibility 1:- same painter paints both blocks, time taken = 55 units
    Possibility 2:- Painter 1 paints block 1, painter 2 paints block 2, 
    time take = max(5, 50) = 50
    There are no other distinct ways to paint boards.
    ans = 50%10000003

Input 2:
    A = 10
    B = 1
    C = [1, 8, 11, 3]
Output 2:
    11

// 118 ms

bool isPossible(int A, int B, vector<int> &C,long long int X){
    int n=C.size();
   long long int t=X;
    int i=0,cnt=1;
    while(i<n){
        if(cnt>A)
         return false;
        if(C[i]>t){
            cnt++;
            t=X;
        }
        else{
            t=t-C[i];
            i++;
        }
    }
    return true;
}


int Solution::paint(int A, int B, vector<int> &C) {
    int n=C.size();
    long long int sum=0;
    for(int i=0;i<n;i++)
     sum=sum%10000003+C[i]%10000003;
    long long int low=0,high=sum*B;
    long long int ans=high%10000003;
    while(low<=high){
        long long int mid=low+(high-low)/2;
        if(isPossible(A,B,C,mid/B)){
            ans=mid%10000003;
            high=mid-1;
        }
        else low=mid+1;
    }
    return ans%10000003;
}

// Editorial , question is good but the platform is merciless , 108 ms

int Solution::paint(int A, int B, vector<int> &C) {
    long long min_time = 0,max_time=0;
    for(int i=0;i<C.size();++i){
        min_time=max(min_time,(long long)C[i]);
        max_time+=(long long)C[i];
    }
    
    assert(B<=1000000);
    long long mid;
    int required=0, time=0;
    while(min_time<=max_time){
        required=1;
        time=0;
        mid = (min_time+max_time)/2;
        for(int i=0; i<C.size(); i++){
            if(time+C[i]>mid){
                time=C[i];
                required++;
            }
            else time+=C[i];
        }
        if(required>A) min_time = mid+1;
        else max_time = mid-1;
    }
    return (B*min_time)% 10000003;
}
