229. Majority Element II
=============================
// Given an integer array of size n, find all elements that appear more than 
// floor(n/3)  times.

 

// Example 1:

// Input: nums = [3,2,3]
// Output: [3]

// Example 2:

// Input: nums = [1]
// Output: [1]

//Boyer–Moore majority vote algorithm O(N), O(1)

vector<int> majorityElement(vector<int> &a) 
  {
	  int num1(-1), num2(-1), cnt1(0), cnt2(0);
    
	  for (const auto & x: a) 
	  {
		  if (x == num1) cnt1++;
		  else if (x == num2) cnt2++;
		  else if (!cnt1) num1 = x, cnt1 = 1;
		  else if (!cnt2) num2 = x, cnt2 = 1;
		  else cnt1--, cnt2--;
     }
      
     cnt1 = cnt2 = 0;
     for (const auto & x: a)
		 if (x == num1) cnt1++;
			 else if (x == num2) cnt2++;
  
	  vector<int> r;
	  if (cnt1 > size(a)/3) r.push_back(num1);
	  if (cnt2 > size(a)/3) r.push_back(num2);
	  return r;
  }


// // Freq(ME) > floor(n/k) , Majority element General
  

// Case 1: Voting for President = Freq(Majority Element) > Floor(N/2)


// If N = 10 => Freq(ME) > 10/2 = 5
// In this case we can only have ONE person (President of the Organisation) getting 
// majority votes. That person can either get 6,7,8,9,10 votes to be called President.

// 7 7 5 7 5 1 5 7 5 5 7 7 5 5 5 5 ---> name of voters
// 1 2 1 2 1 0 1 0 1 2 1 0 1 2 3 4 ---> cnt of win margin

// Fianlly 5 is the ME.

// // count == 0, majority and minority exhausted so next new element is made as ME

// Case 2: Voting for President and Vice President = Freq(Majority Element) > Floor(N/3)

// If N = 15 => Freq(ME) > 15/3 = 5
// In this case realise that we can 3 cases.
// i. 0 people winning: everyone gets unique votes or some gets 2 etc
// ii. 1 person winning: President who gets vote like 6 and rest all get less than 6.
// Example: 1 1 1 1 1 1 2 2 3 3 4 4 5 5 6
// President = 1 with 6 votes.
// iii. 2 people winning: President and Vice President both get equal to 6 or one 
// of them gets more than 6.
// Example: 1 1 1 1 1 1 1 2 2 2 2 2 2 8 8
// President = 1 with 7 votes.
// Vice President = 2 with 6 votes.
// Rest = 2 votes.

// when we hit a minority element we decrease both cnt as it is affecting winning 
// difference of both the winning Candidates. We want our counters to be greater 
// than 1 for that candidate to win.

// index:0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
// arr:....1 2 3 1 1 1 7 7 1 1 7 5 7 7 7
// cnt1: 1....0 1 2 3........4 5....4
// cnt2:....1.0..........1 2........3 2 3 4 5

// Case 3: Voting for President, Sr. Vice President and Vice 
// President = Freq(Majority Element) > Floor(N/4)

// If N = 20 => Freq(ME) > 20/4 = 5
// In this case realise that we can have 4 cases. 0,1,2 or 3 positions. Must 
// have realised it till now.

// Major Takeaways
// -----------------------
//  (1)Visualise from voting point of view and how a winning candidate actually wins 
//      becuase his/her votes difference from Minority Elements(not winning candidates) 
//      is greater than 1.
//  (2)Till now you must have realised for a question where Freq(ME) > N/k we need k - 1 
//      counters. OR For Example if Freq(ME) > N/4 we need 3 counters as there is 
//      battle for max 3 positions.
//  (3)When you arrive at an element which neither matches any of the candidates 
//      i.e (Majority Elements) that particular element or vote is a equal loss to 
//      every winning candidate, hence all the counters must be decremented by 1.


vector<int> majorityElementGeneralised(vector<int> arr, int k) {
   int n = arr.size();

   // Map of Majority Element(ME), Count
   // Count is not Frequency in array.
   // For checking Freq(ME) > floor(n/k) we run another loop
   // on counters.first element or on ME's
   map<int, int> mp;

   for (auto x : arr) {
      if (mp.count(x)) ++mp[x]; //increase count if x present
      else if(mp.size()<k-1) ++mp[x]; // Freq(ME) > n/k then k-1 counter possible
      else {    
         for(auto it = mp.begin();it != mp.end();) {
            int key = it->first;
            if (it->second == 1) { 
               //counter == 0,remove.if it becomes ME will be initialised with 1 
               it = mp.erase(it); //if winning margin for 1 is = 5 : 1 1 1 1 1
            }
            else {
               --mp[key]; // we have new potential ME candidate/Minority exhausted
               ++it;
            }
         } // end of inner for loop
      }
   } // end of outer for loop

   int validFreq = floor(n / k);

   vector<int> result = {};
   //k=4,a=[1, 2, 3, 1, 1, 1, 7, 7, 1, 1, 7, 5, 7, 7, 7],mp={1:1-1+3+2=5,5:1,7:2+1+1=5}
   //a=[1, 2, 5, 1, 1, 2, 7, 7, 1, 1, 7, 5, 7, 7, 7],mp={1:1-1+2-1+2=3,2:1-1=0,7:2-1+3=4} --> 2 removed
   // for 1st case else condition for key=1 & for second key=1,2,1,7 --> order of appearance 
   for (auto x : arr) { // Make ME freq = 0
      if (mp.count(x))
         mp[x] = 0;
   }

   //a=[1, 2, 5, 1, 1, 2, 7, 7, 1, 1, 7, 5, 7, 7, 7],mp={1:5,7:6}
   //a=[1, 2, 3, 1, 1, 1, 7, 7, 1, 1, 7, 5, 7, 7, 7],mp={1:6,5:1,7:6}
   for (auto x : arr) {
      if (mp.count(x))
         ++mp[x];
   }

   // Check is valid Majority Element
   for (auto x : mp)
      if (x.second > validFreq)
         result.push_back(x.first);

   return result;
}