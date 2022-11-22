Buy Maximum Stocks if i stocks can be bought on i-th day
=============================================================
// In a stock market, there is a product with its infinite stocks. 
// The stock prices are given for N days, where arr[i] denotes the price of the 
// stock on the ith day. There is a rule that a customer can buy at most i stock 
// on the ith day. If the customer has an amount of k amount of money initially, 
// find out the maximum number of stocks a customer can buy. 

// For example, for 3 days the price of a stock is given as 7, 10, 4. 
// You can buy 1 stock worth 7 rs on day 1, 2 stocks worth 10 rs each on day 2 
// and 3 stock worth 4 rs each on day 3.

// Examples: 

// Input : price[] = { 10, 7, 19 }, 
//               k = 45.
// Output : 4
// A customer purchases 1 stock on day 1, 
// 2 stocks on day 2 and 1 stock on day 3 for 
// 10, 7 * 2 = 14 and 19 respectively. Hence, 
// total amount is 10 + 14 + 19 = 43 and number 
// of stocks purchased is 4.

// Input  : price[] = { 7, 10, 4 }, 
//                k = 100.
// Output : 6

// Let say, we have R rs remaining till now, and the cost of product on this day 
// be C, and we can buy atmost L products on this day then, 
// total purchase on this day (P) = min(L, R/C) 


// Return the maximum stocks
int buyMaximumProducts(int n, int k, int price[])
{
    vector<pair<int, int> > v; // <C,L>
 
    // Making pair of product cost and number
    // of day..
    for (int i = 0; i < n; ++i)
        v.push_back(make_pair(price[i], i + 1));   
 
    // Sorting the vector pair.
    sort(v.begin(), v.end());   
 
    // Calculating the maximum number of stock
    // count.
    int ans = 0;
    for (int i = 0; i < n; ++i) {
        int l = v[i].second;
        int c = v[i].first;
        int p = min(l, k/c); // total purchase on that day
        ans += p
        k -= c * p; 
        //we try to decrease k as slowly as possible taking as many number as 
        //possible
    }
 
    return ans;
}