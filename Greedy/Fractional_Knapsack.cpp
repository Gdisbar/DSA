Fractional Knapsack 
======================
// Given weights and values of N items, we need to put these items in a 
// knapsack of capacity W to get the maximum total value in the knapsack.
// Note: Unlike 0/1 knapsack, you are allowed to break the item. 

 

// Example 1:

// Input:
// N = 3, W = 50
// values[] = {60,100,120}
// weight[] = {10,20,30}
// Output:
// 240.00
// Explanation:Total maximum value of item
// we can have is 240.00 from the given
// capacity of sack. 

// Example 2:

// Input:
// N = 2, W = 50
// values[] = {60,100}
// weight[] = {10,20}
// Output:
// 160.00
// Explanation:
// Total maximum value of item
// we can have is 160.00 from the given
// capacity of sack.

// TC : n*log(n) , SC : 1


// Structure for an item which stores weight and
// corresponding value of Item
struct Item {
    int value, weight;
    //Item(x,w) : value(x),weight(w);
    // Constructor
    Item(int value, int weight)
    {
        this->value = value;
        this->weight = weight;
    }
};
 
// Comparison function to sort Item according to val/weight
// ratio
bool cmp(struct Item a, struct Item b)
{
    double r1 = (double)a.value / (double)a.weight;
    double r2 = (double)b.value / (double)b.weight;
    return r1 > r2;
}
 
// Main greedy function to solve problem
double fractionalKnapsack(int W, struct Item arr[], int n)
{
    //    sorting Item on basis of ratio
    sort(arr, arr + n, cmp);
 
    double finalvalue = 0.0; // Result (value in Knapsack)
 
    // Looping through all Items
    for (int i = 0; i < n; i++) {
        // If adding Item won't overflow, add it completely
        if (arr[i].weight <= W) {
            W -= arr[i].weight;
            finalvalue += arr[i].value;
        }
 
        // If we can't add current Item, add fractional part
        // of it ---> W/w[i] , here w[i] > W
        else {
            finalvalue+= arr[i].value* ((double)W / (double)arr[i].weight);
            break;
        }
    }
 
    // Returning final value
    return finalvalue;
}
 
// TC : n (each item is pushed at the beginning) , SC : n

struct Item{
    int value;
    int weight;
};

class Solution
{
    public:
    struct compare1{
        bool operator()(Item it1,Item it2){
          return (double)it2.value/it2.weight > (double)it1.value/it1.weight;
       }
    };
    
    //Function to get the maximum total value in the knapsack.
    double fractionalKnapsack(int W, Item arr[], int n)
    {
        // Your code here
        priority_queue<Item,vector<Item>,compare1> pq;
        for(int i=0;i<n;++i) pq.push(arr[i]);
        double ans=0;
        while(W&&!pq.empty()){
            auto tmp=pq.top();
            pq.pop();
            if(tmp.weight<=W){
                ans+=tmp.value;
                W-=tmp.weight;
            }
            else{
                ans+=(double) tmp.value/tmp.weight*W;
                W=0;
            }
        }
        return ans;
    }
        
};

// DP version

bool sortbySec(const pair<int,double>&a,const pair<int,double>&b){
    return a.second > b.second;
}
vector<pair<int,double>> priceEvaluator(vector<int>&w,vector<int>&p){
    vector<pair<int, double>>wp;
    for (int i{0}; i < w.size();i++){
        wp.push_back(make_pair(w[i],p[i] * 1.0/w[i]));
    }
    sort(wp.begin(),wp.end(),sortbySec);
    return wp;
}
float fractionalKnapsack(vector<pair<int,double>>&wp,int capacity){
    vector<double> dp(capacity+1,0);
    for (int i{1}; i <= capacity;i++){
        int cap = i;
        for (int j{0}; j < wp.size() && cap > 0;j++){
            dp[i] += min({cap,wp[j].first}) * wp[j].second;
            cap -= min({cap,wp[j].first});
        }
    }
    return dp[capacity];
}