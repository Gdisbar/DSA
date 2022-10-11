Buddy NIM | Problem Code:BUDDYNIM
=====================================
// Alice, Bob and Charlie are playing a new game called Buddy NIM. The game is 
// played at two tables; on the first table, there are N heaps containing 
// A1,A2,…,AN stones and on the second table, there are M heaps containing 
// B1,B2,…,BM stones respectively.

// Initially, Alice is playing at the first table and Bob is playing at the second 
// table. The players take their turns in this order: Charlie, Alice, Bob, 
// Charlie, etc.

// Alice and Bob follow the rules for classical NIM - on Alice's turn, Alice must 
// remove a positive number of stones from one of the piles at her current table 
// and on Bob's turn, Bob must remove a positive number of stones from one of the 
// piles at his current table. Whoever cannot remove any stone from a pile loses.

// Charlie does not play at any table. Instead, on his turn, he decides if Alice 
// and Bob should keep playing at their respective tables or swap places.

// Alice and Charlie are buddies and they cooperate, playing in the optimal way 
// that results in Alice''s victory (if possible).

// It is clear that either Alice or Bob wins the game eventually. You must find out 
// who the winner will be.


// Example Input

// 3 --> test case

// 3 1 --> n m
// 1 1 1 --> table 1
// 3     --> table 2

// 3 1
// 1 2 4
// 7
// 1 1
// 1
// 1

// Example Output

// Alice
// Alice
// Bob


Alice : a1,a2,a3,...an , Bob : b1,b2,...,bm
// calculate for all non-zero
case 1: Σa[i] != Σb[i] 
//for Alice to win Charlie has to give table with max stone to Alice
		1. Σa[i] > Σb[i] => no switch 
		2. else switch
case 2: Σa[i] == Σb[i] 
//for Bob to win configaration has to be same on both the table
		1. after k turns Σa[i]-k1 = Σb[i]-k2 
	//Bob win if k1==k2 but if k1>k2 or k2<k1 Charlie will swap & Alice win : case-1
	//but if k1==k2==k the turn goes to Alice [Alice,Bob,Alice,Bob]--> Alice
	//if (n==m && a[i]==b[j]) otherwise Alice will remove a large no of pile
	// and if Bob can't remove the same no Bob will loose

// TC : nlogn+mlogm

void solve(){
   int n,m;cin>>n>>m;
   vi a(n),b(m);
   each(x,a) cin>>x;
   each(x,b) cin>>x;
   map<int,int> alice,bob;
   each(x,a){
    if(x!=0)
        alice[x]++;
   } 
   each(x,b){
    if(x!=0)
        bob[x]++;
   }
   if(alice==bob) cout<<"Bob";
   else cout<<"Alice";
}