Mode of Frequencies | Problem Code:MODEFREQ
===============================================
// There are N numbers in a list A=A1,A2,…,AN. Chef needs to find the mode of the 
// frequencies of the numbers. If there are multiple modal values, report the 
// smallest one. In other words, find the frequency of all the numbers, and 
// then find the frequency which has the highest frequency. If multiple such 
// frequencies exist, report the smallest (non-zero) one.

// More formally, for every v such that there exists at least one i such 
// that Ai=v, find the number of j such that Aj=v, and call that the frequency 
// of v, denoted by freq(v). Then find the value www such that freq(v)=w for 
// the most number of vconsidered in the previous step. If there are multiple 
// values w which satisfy this, output the smallest among them.

// Input :
// 2 --> T
// 8 ---> N
// 5 9 2 9 7 2 5 3
// 9
// 5 9 2 9 7 2 5 3 1

// Output:
// 2
// 1

// Explanation:

// Test case 1: (2, 9 and 5) have frequency 2, while (3 and 7)have frequency 1. 
// Three numbers have frequency 2, while 2 numbers have frequency 1. Thus, the mode 
// of the frequencies is 2.

// Test case 2: (2, 9 and 5) have frequency 2, while (3, 1 and 7) have frequency 1. 
// Three numbers have frequency 2, and 3 numbers have frequency 1. Since there are 
// two modal values 1 and 2, we report the smaller one: 1.

int main()
{
    int T, N;
    cin >> T;
    while (T--)
    {
        cin >> N;
        int A[N];
        int B[10] = {0};
        for (int i = 0; i < N; i++)
        {
            cin >> A[i];
        }
        for (int j = 0; j < N; j++)
        {
        	//0-based index for digits 1-10 ,(2, 9 and 5) have frequency 2
            B[A[j] - 1]++; 
        }
        int C[10001] = {0};
        for (int i = 0; i < 10; i++)
        {
            if (B[i] > 0)
            {
            	//frequency of same frequency value,3 numbers have frequency 2
                C[B[i] - 1]++; 
            }
        }
        int max_num = INT_MIN;
        for (int i = 0; i < 10001; i++)
        {
            max_num = max(max_num, C[i]);
        }
        for (int i = 0; i < 10001; i++)
        {
            if (C[i] == max_num)
            {
                cout << i + 1 << endl;
                break;
            }
        }
    }
    return 0;
}