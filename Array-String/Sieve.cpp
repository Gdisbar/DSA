//prime test for large number of queries (1000000) --> preprocess
// preprocess : O(nlog(log(n))),query : O(1),space : O(n)

//global array is initialized by 0
int is_prime[1000001];

void sieve(){
	int N = 1000000;
	for(int i = 1;i<=N;i++) is_prime[i]=1;
	is_prime[0]=is_prime[1]=0;
    
    for(int i = 2;i*i<=N;i++){
    	if(is_prime[i]){
    		for(int j = i*i;j<=N;j+=i)is_prime[j]=0;
    	}
    }
}


// Segmented Sieve
PRIME1 - Prime Generator
============================
//  Peter wants to generate some prime numbers for his cryptosystem. 
//  Help him! Your task is to generate all prime numbers between two given numbers!



// Input:
// 2      // t
// 1 10   // l,r
// 3 5

// Output: 
// // prime in range [l=1,r=10]
// 2
// 3
// 5
// 7

// 3
// 5


//Without pre-generating all prime numbers:
//O((R-L+1)loglog(R)+sqrt(R)loglog(sqrt(R)))

vector<bool> segmentedSieveNoPreGen(long long L, long long R) {
    vector<bool> isPrime(R - L + 1, true);
    long long lim = sqrt(R);
    for (long long i = 2; i <= lim; ++i)
        for (long long j = max(i * i, (L + i - 1) / i * i); j <= R; j += i)
            isPrime[j - L] = false;
    if (L == 1)
        isPrime[0] = false;
    return isPrime; //primes are --> L+i 
}

//With pre-generating
//O((R-L+1)log(R)+sqrt(R))

vector<int> primes;

void sieve(int N){
	vector<int> is_prime(N+1,1);
	is_prime[0]= is_prime[1]=0;

	for(int i = 2;i*i<=N;i++)
	  if(is_prime[i]==1){
	  	for(int j = 2*i;j<=N;j+=i)
	  		is_prime[j]=0;
	  } 
    
    for(int i = 1;i*i<=N;i++){
    	if(is_prime[i]==1){
    		primes.push_back(i);
    	}
    }
}

void init(int L,int R){
	if(L==1) L++;
	int N = R - L +1;
	sieve(N);
	for(auto &p : primes){
		if(p*p<=R){
			//For example, if L = 31 and p = 3, we start with i = 33.
			int i = (L/p)*p;
			if(i<L) i += p;
			//i --> 1st multiple of p in range(L,R)
			for(;i<=R;i+=p){
				if(i!=p)
					isPrime[i-L]=0;
			}
		}
	}

	for(int i = 0;i<N;i++)
		if(isPrime[i]==1)
			cout<<L+i<<endl;
}