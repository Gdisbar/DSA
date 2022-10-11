B. Napoleon Cake # Codeforces Round #707 (Div. 2, based on Moscow Open Olympiad in Informatics)
====================================================================================================
// To bake a Napoleon cake, one has to bake n dry layers first, and then put them 
// on each other in one stack, adding some cream. Arkady started with an empty 
// plate, and performed the following steps n times:

//     place a new cake layer on the top of the stack;
//     after the i-th layer is placed, pour a[i] units of cream on top of the stack. 

// When x units of cream are poured on the top of the stack, top x layers of 
// the cake get drenched in the cream. If there are less than x layers, all layers 
// get drenched and the rest of the cream is wasted. If x=0, no layer gets drenched.
// The picture represents the first test case of the example.

// Help Arkady determine which layers of the cake eventually get drenched when 
// the process is over, and which don''t.


// start from right to left of the array (backward) & apply cream --> let the cream dip
// in lower layers

// a=     0 3 | 0 | 0 1 3  
// s=     1 1 | 0 | 1 1 1 

//a= 0 |0 0 1 0 5| 0 0 |0 2
//s= 0 |1 1 1 1 1| 0 0 |1 1 


stack<int> s;
int cur =a[n-1];
repi(i,n-1,0){
  if(a[i]>=cur)cur=a[i];
  if(cur>0&&cur>=a[i]){
    cur--;
    s.push(1);
  }
  else s.push(0);
}