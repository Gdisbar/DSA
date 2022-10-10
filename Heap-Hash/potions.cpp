C2. Potions (Hard Version) # Codeforces Round #723 (Div. 2)
===============================================================
// There are n potions in a line, with potion 1 on the far left and potion n on the 
// far right. Each potion will increase your health by ai when drunk. a[i] can be 
// negative, meaning that potion will decrease will health.

// You start with 0 health and you will walk from left to right, from first potion 
// to the last one. At each potion, you may choose to drink it or ignore it. You must 
// ensure that your health is always non-negative.

// What is the largest number of potions you can drink?


// We process the potions from left to right. At the same time, we maintain the 
// list of potions we have taken so far. When processing potion i, if we can take i 
// without dying, then we take it. Otherwise, if the most negative potion we''ve taken is 
// more negative than potion i, then we can swap out potion i for that potion. To find 
// the most negative potion we''ve taken, we can maintain the values of all potions in a 
// minimum priority_queue. 

// This runs in O(nlogn) as well

// To prove that this works, let''s consider best solution where we take exactly k
// potions (best as in max total health). The solution involves taking the k

// largest values in our priority queue. Then when considering a new potion, 
// we should see whether swapping out the new potion for the $k$th largest potion 
// will improve the answer.

// Since the priority queue is strictly decreasing, there will be a cutoff K, where for 
// k at most K, the answer is not affected, and for larger than K, we swap out the kth 
// largest potion. It turns out this process is equivalent to inserting the new 
// potion''s value into the priority_queue. For those positions at most K, they are not 
// affected. For the positions larger than K, the elements get pushed back one space, 
// meaning that the smallest element is undrinked.

// This can also be seen as an efficient way to to transition from one layer of the dp
// table to the next.

// Input :
// 6
// 4 -4 1 -3 1 -3

// Output : 5

// Note
// -------
// For the sample, you can drink 5 potions by taking potions 1, 3, 4, 5 and 6. 
// It is not possible to drink all 6 potions because your health will go negative at 
// some point



void solve(){
    ll n;cin>>n;
    priority_queue<ll,vector<ll>,greater<ll>> q;
    ll total=0;
    rep(i,0,n-1){
        ll x;cin>>x;
        total+=x;
        q.push(x);
        while(total<0){
            total-=q.top();
            q.pop();
        }
    }
    cout<<(int)q.size();
}