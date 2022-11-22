191. Number of 1 Bits
========================
// Write a function that takes an unsigned integer and returns the number of '1' 
// bits it has (also known as the Hamming weight).

// Note:

//     Note that in some languages, such as Java, there is no unsigned integer type. 
//     In this case, the input will be given as a signed integer type. 
//     It should not affect your implementation, as the integer's internal binary 
//     representation is the same, whether it is signed or unsigned.
//     In Java, the compiler represents the signed integers using 2's complement 
//     notation. Therefore, in Example 3, the input represents the signed integer. -3.

 

// Example 1:

// Input: n = 00000000000000000000000000001011
// Output: 3
// Explanation: The input binary string 00000000000000000000000000001011 
// has a total of three '1' bits.

// Example 2:

// Input: n = 00000000000000000000000010000000
// Output: 1
// Explanation: The input binary string 00000000000000000000000010000000 
// has a total of one '1' bit.

// Example 3:

// Input: n = 11111111111111111111111111111101
// Output: 31
// Explanation: The input binary string 11111111111111111111111111111101 
// has a total of thirty one '1' bits.


int hammingWeight(uint32_t n) {
    int count = 0;
    
    while (n) {
        n &= (n - 1);
        count++;
    }
    
    return count;
}

// n & (n - 1) drops the lowest set bit. It's a neat little bit trick.

// Let's use n = 00101100 as an example. This binary representation has three 1s.

// If n = 00101100, then n - 1 = 00101011, 
// so n & (n - 1) = 00101100 & 00101011 = 00101000. Count = 1.

// If n = 00101000, then n - 1 = 00100111, 
// so n & (n - 1) = 00101000 & 00100111 = 00100000. Count = 2.

// If n = 00100000, then n - 1 = 00011111, 
// so n & (n - 1) = 00100000 & 00011111 = 00000000. Count = 3.

// n is now zero, so the while loop ends, and the final count 
// (the numbers of set bits) is returned.



190. Reverse Bits
==================
// Reverse bits of a given 32 bits unsigned integer.

// Note:

//     Note that in some languages, such as Java, there is no unsigned integer type. 
//     In this case, both input and output will be given as a signed integer type. 
//     They should not affect your implementation, as the integer's internal binary 
//     representation is the same, whether it is signed or unsigned.
//     In Java, the compiler represents the signed integers using 2's complement 
//     notation. Therefore, in Example 2 above, the input represents the signed 
//     integer -3 and the output represents the signed integer -1073741825.

 

// Example 1:

// Input: n = 00000010100101000001111010011100
// Output:    964176192 (00111001011110000010100101000000)
// Explanation: The input binary string 00000010100101000001111010011100 
// represents the unsigned integer 43261596, so return 964176192 which its 
// binary representation is 00111001011110000010100101000000.

// Example 2:

// Input: n = 11111111111111111111111111111101
// Output:   3221225471 (10111111111111111111111111111111)
// Explanation: The input binary string 11111111111111111111111111111101 
// represents the unsigned integer 4294967293, so return 3221225471 which its 
// binary representation is 10111111111111111111111111111111.

// for 8 bit binary number abcdefgh, the process is as follow:

// abcdefgh -> efghabcd -> ghefcdab -> hgfedcba

// this algorithm swaps the bits in the following steps:

//     1. 16 bits left and right swapped
//     2. every couple of 8 bits swapped (every other 8 bits are picked by AND 
//     	operation and 00ff and ff00 as masks equivalent to 0000000011111111 
//     	and 1111111100000000)
//     3. every couple of 4 bits are swapped like above using 0f0f and f0f0 as masks.
//     4. every couple of 2 bits are swapped using cc and 33 corresponding to 
//        11001100 and 0011011
//     5. every couple of 1 bit are swapped using aa and 55 corresponding to 
//        10101010 and 01010101


//     This results in log(D) time complexity in which D is the number of bits.

// Step 0.
// abcd efgh ijkl mnop qrst uvwx yzAB CDEF <-- n

// Step 1.
//                     abcd efgh ijkl mnop <-- n >> 16, same as (n & 0xffff0000) >> 16
// qrst uvwx yzAB CDEF                     <-- n << 16, same as (n & 0x0000ffff) << 16
// qrst uvwx yzAB CDEF abcd efgh ijkl mnop <-- after OR

// Step 2.
//           qrst uvwx           abcd efgh <-- (n & 0xff00ff00) >> 8
// yzAB CDEF           ijkl mnop           <-- (n & 0x00ff00ff) << 8
// yzAB CDEF qrst uvwx ijkl mnop abcd efgh <-- after OR

// Step 3.
//      yzAB      qrst      ijkl      abcd <-- (n & 0xf0f0f0f0) >> 4
// CDEF      uvwx      mnop      efgh      <-- (n & 0x0f0f0f0f) << 4
// CDEF yzAB uvwx qrst mnop ijkl efgh abcd <-- after OR

// Step 4.
//   CD   yz   uv   qr   mn   ij   ef   ab <-- (n & 0xcccccccc) >> 2
// EF   AB   wx   st   op   kl   gh   cd   <-- (n & 0x33333333) << 2
// EFCD AByz wxuv stqr opmn klij ghef cdab <-- after OR

// Step 5.
//  E C  A y  w u  s q  o m  k i  g e  c a <-- (n & 0xaaaaaaaa) >> 1
// F D  B z  x v  t r  p n  l j  h f  d b  <-- (n & 0x55555555) << 1
// FEDC BAzy xwvu tsrq ponm lkji hgfe dcba <-- after OR

uint32_t reverseBits(uint32_t n) {
        n = (n >> 16) | (n << 16);
        n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8);
        n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4);
        n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2);
        n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1);
        return n;
    }


//Alternate

int reverseBits(int n) {
    if (n == 0) return 0;
    
    int result = 0;
    for (int i = 0; i < 32; i++) {
        result <<= 1;
        if ((n & 1) == 1) result++;
        n >>= 1;
    }
    return result;
}

// We first intitialize result to 0. We then iterate from
// 0 to 31 (an integer has 32 bits). In each iteration:
// We first shift result to the left by 1 bit.
// Then, if the last digit of input n is 1, we add 1 to result. To
// find the last digit of n, we just do: (n & 1)
// Example, if n=5 (101), n&1 = 101 & 001 = 001 = 1;
// however, if n = 2 (10), n&1 = 10 & 01 = 00 = 0).

// Finally, we update n by shifting it to the right by 1 (n >>= 1). This is because the last digit is already taken care of, so we need to drop it by shifting n to the right by 1.

// At the end of the iteration, we return result.

// Example, if input n = 13 (represented in binary as
// 0000_0000_0000_0000_0000_0000_0000_1101, the "_" is for readability),
// calling reverseBits(13) should return:
// 1011_0000_0000_0000_0000_0000_0000_0000

// Here is how our algorithm would work for input n = 13:

// Initially, result = 0 = 0000_0000_0000_0000_0000_0000_0000_0000,
// n = 13 = 0000_0000_0000_0000_0000_0000_0000_1101

// Starting for loop:
// i = 0:
// result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_0000.
// n&1 = 0000_0000_0000_0000_0000_0000_0000_1101
// & 0000_0000_0000_0000_0000_0000_0000_0001
// = 0000_0000_0000_0000_0000_0000_0000_0001 = 1
// therefore result = result + 1 =
// 0000_0000_0000_0000_0000_0000_0000_0000
// + 0000_0000_0000_0000_0000_0000_0000_0001
// = 0000_0000_0000_0000_0000_0000_0000_0001 = 1

// Next, we right shift n by 1 (n >>= 1) (i.e. we drop the least significant bit) to get:
// n = 0000_0000_0000_0000_0000_0000_0000_0110.
// We then go to the next iteration.

// i = 1:
// result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_0010;
// n&1 = 0000_0000_0000_0000_0000_0000_0000_0110 &
// 0000_0000_0000_0000_0000_0000_0000_0001
// = 0000_0000_0000_0000_0000_0000_0000_0000 = 0;
// therefore we don't increment result.
// We right shift n by 1 (n >>= 1) to get:
// n = 0000_0000_0000_0000_0000_0000_0000_0011.
// We then go to the next iteration.

// i = 2:
// result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_0100.
// n&1 = 0000_0000_0000_0000_0000_0000_0000_0011 &
// 0000_0000_0000_0000_0000_0000_0000_0001 =
// 0000_0000_0000_0000_0000_0000_0000_0001 = 1
// therefore result = result + 1 =
// 0000_0000_0000_0000_0000_0000_0000_0100 +
// 0000_0000_0000_0000_0000_0000_0000_0001 =
// result = 0000_0000_0000_0000_0000_0000_0000_0101
// We right shift n by 1 to get:
// n = 0000_0000_0000_0000_0000_0000_0000_0001.
// We then go to the next iteration.

// i = 3:
// result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_1010.
// n&1 = 0000_0000_0000_0000_0000_0000_0000_0001 &
// 0000_0000_0000_0000_0000_0000_0000_0001 =
// 0000_0000_0000_0000_0000_0000_0000_0001 = 1
// therefore result = result + 1 =
// = 0000_0000_0000_0000_0000_0000_0000_1011
// We right shift n by 1 to get:
// n = 0000_0000_0000_0000_0000_0000_0000_0000 = 0.

// Now, from here to the end of the iteration, n is 0, so (n&1)
// will always be 0 and and n >>=1 will not change n. The only change
// will be for result <<=1, i.e. shifting result to the left by 1 digit.
// Since there we have i=4 to i = 31 iterations left, this will result
// in padding 28 0's to the right of result. i.e at the end, we get
// result = 1011_0000_0000_0000_0000_0000_0000_0000