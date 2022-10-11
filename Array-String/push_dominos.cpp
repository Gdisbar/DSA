838. Push Dominoes
=====================
// There are n dominoes in a line, and we place each domino vertically upright. 
// In the beginning, we simultaneously push some of the dominoes either to the left 
// or to the right.

// After each second, each domino that is falling to the left pushes the adjacent 
// domino on the left. Similarly, the dominoes falling to the right push their 
// adjacent dominoes standing on the right.

// When a vertical domino has dominoes falling on it from both sides, it stays 
// still due to the balance of the forces.

// For the purposes of this question, we will consider that a falling domino 
// expends no additional force to a falling or already fallen domino.

// You are given a string dominoes representing the initial state where:

//     dominoes[i] = 'L', if the ith domino has been pushed to the left,
//     dominoes[i] = 'R', if the ith domino has been pushed to the right, and
//     dominoes[i] = '.', if the ith domino has not been pushed.

// Return a string representing the final state.

 

// Example 1:

// Input: dominoes = "RR.L"
// Output: "RR.L"
// Explanation: The first domino expends no additional force on the second domino.

// Example 2:

// Input: dominoes = ".L.R...LR..L.."
// Output: "LL.RR.LLRRLL.."

/*
	In this approach, you just need to find sections like this
	X .   .   .   . X
	i                j
	Where X can be 'R' or 'L' and in between there can be as many dots
	Now,
	- you know the length of mid part
	- If char[i] == char[j] == 'R', means all go towards right (R)
	-  char[i]  == char[j] == 'L', means all go towards Left (L)
	-  If char[i] = 'L' and char[j] = 'R', means middle part is not affected so they 
		remain '.'
	-  If char[i] = 'R' and char[j] = 'L', then it will affect the middle part.
	   The middle_part/2 close to i will be affected by 'R' and middle_part/2 
	   close to j will be effected by 'L'  and the last mid point (middle_part%2) 
	   will be unaffected due to equal force from left and right so it remains '.'
*/

string pushDominoes(string d) {
        d = 'L' + d + 'R';
        string res = "";
        for (int i = 0, j = 1; j < d.length(); ++j) {
            if (d[j] == '.') continue;
            int middle = j - i - 1;
            if (i > 0)
                res += d[i];
            if (d[i] == d[j])
                res += string(middle, d[i]);
            else if (d[i] == 'L' && d[j] == 'R')
                res += string(middle, '.');
            else
                res += string(middle / 2, 'R') + string(middle % 2, '.') + string(middle / 2, 'L');
            i = j;
        }
        return res;
    }

def pushDominoes(self, d):
        d = 'L' + d + 'R'
        res = ""
        i = 0
        for j in range(1, len(d)):
            if d[j] == '.':
                continue
            middle = j - i - 1
            if i:
                res += d[i]
            if d[i] == d[j]:
                res += d[i] * middle
            elif d[i] == 'L' and d[j] == 'R':
                res += '.' * middle
            else:
                res += 'R' * (middle / 2) + '.' * (middle % 2) + 'L' * (middle / 2)
            i = j
        return res

# https://www.youtube.com/watch?v=evUFsOb_iLY
        
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        dom=list(dominoes)
        q=deque()
        for i,d in enumerate(dom):
            if d!=".": q.append((i,d))
        
        while q:
            i,d = q.popleft()
            if d=="L":
                if i > 0 and dom[i-1]==".":
                    q.append((i-1,"L"))
                    dom[i-1]="L"
            elif d=="R":
                if i+1<len(dom) and dom[i+1]==".":
                    if i+2<len(dom) and dom[i+2]=="L":
                        q.popleft()
                    else:
                        q.append((i+1,"R"))
                        dom[i+1]="R"
        
        return "".join(map(str,dom))