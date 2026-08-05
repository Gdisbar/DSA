
## Pattern 1 — Balanced Symbols

| Section | Details |
|---|---|
| Recognition | () [] {} • Nested • Balanced • Valid • HTML/XML Tags |
| Core Idea | Top of stack must match the current closing symbol — tag matching is the same idea with word-tokens instead of brackets. |
| Time / Space | O(n) / O(n) |
| Common Trap | Compare only with the top, never the bottom. |

---

### 20. Valid Parentheses

Given a string s containing just the characters ``'(', ')', '{', '}', '[' and ']'``, determine if the input string is valid.

An input string is valid if:

- Open brackets must be closed by the same type of brackets.
- Open brackets must be closed in the correct order.
- Every close bracket has a corresponding open bracket of the same type.
 

**Example 1**:

    Input: s = "([])"
    Output: true

**Example 2**:

    Input: s = "([)]"
    Output: false

```python

def isValid(self, s: str) -> bool:
    i=0
    a=[]
    for i in range(len(s)):
        if s[i]=='('or s[i]=='['or s[i]=='{':
            a.append(s[i])
        else:
            # stack is empty 
            if not a:
                return False
            top=a.pop()
            if s[i]==')'and top!='(':
                return False
            if s[i]==']'and top!='[':
                return False
            if s[i]=='}'and top!='{':
                return False
    return len(a)==0

# java

public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    for (char ch : s.toCharArray()) {
        if (ch == '(' || ch == '[' || ch == '{') {
            stack.push(ch);
        } else {
            if (stack.isEmpty()) {
                return false;
            }
            char top = stack.pop();
            if (ch == ')' && top != '(') {
                return false;
            }
            if (ch == ']' && top != '[') {
                return false;
            }
            if (ch == '}' && top != '{') {
                return false;
            }
        }
    }
    return stack.isEmpty();
}
```

### 32. Longest Valid Parentheses

Given a string containing just the characters '(' and ')', *return the length of the longest valid (well-formed) parentheses substring*.

**Example 1**:

    Input: s = "(()"
    Output: 2
    Explanation: The longest valid parentheses substring is "()".

**Example 2**:

    Input: s = ")()())"
    Output: 4
    Explanation: The longest valid parentheses substring is "()()".



**case-1.** ``s[i]='('`` : push to stack

**case-2.**  ``s[i]=')'``: pop the index from the stack (again just as in parentheses check). Again, there will be following scenarios that may occur

**case-2.1** : *stack is not empty* - If stack is not empty, then this may be our longest valid parentheses. We update the MAX_len. 

- Do notice, that our bottom of stack will always hold index preceding to a potential valid parentheses.

**case-2.2** : *stack becomes empty* - This will only happen when we have an extra ``')'`` bracket. There may have been valid parentheses previously which have been updated and stored in MAX_len. But, since we now have an extra closing bracket any further extensions of previous valid parentheses is not possible. 

- So, push the current index into stack, again which will denote that bottom of stack will hold the index preceding to a potential valid parentheses

```bash

Example - '()())()'
Initial stack(from bottom to top) :  [ -1 ] , MAX = 0

1. i = 0          |   s[i] = '('        =>     case-1: push current index into stack
stack : [-1, 0]   |   MAX = 0

2. i = 1          |   s[i] = ')'        =>     case-2.1: pop. After pop, stack is not empty so update MAX.
stack : [-1]      |   MAX = max(0, 1 - (-1)) = 2.

'NOTE : Since the index starts from 0, having index preceding to the start of valid parentheses will give us actual length of the valid parentheses,
instead of us having to add 1 to it everytime.'

3. i = 2          |   s[i] = '('        =>     case-1: push current index into stack
stack : [-1, 1]   |   MAX = 2.

4. i = 3          |   s[i] = ')'        =>     case-2.1: pop. After pop, stack is not empty so update MAX.
stack : [-1]      |   MAX = max(2, 3 - (-1)) = 4.

5. i = 4          |   s[i] = ')'        =>     case-2.2: pop. After pop, stack is empty, so push current index into stack.
This denotes any valid parentheses from now will start from next index and previous valid parentheses cant be extended further.
stack : [4]       |   MAX = 4.

6. i = 5          |   s[i] = '('        =>     case-1: push current index into stack
stack : [4, 5]    |   MAX = 4.

7. i = 6          |   s[i] = ')'        =>     case-2.2: pop. After pop, stack is empty, so push current index into stack.
stack : [4]       |   MAX = max(4, 6 - 4) = 4.
```

```python

def longestValidParentheses(self, s: str) -> int:
    # bottom of stack will always hold index 
    # preceding to potential start of valid parentheses
    stack = [-1]
    max_len = 0

    for i in range(len(s)):
        if s[i] == "(":
            stack.append(i)
        else:
            stack.pop()
            # stack empty - case-2.2
            if len(stack) == 0:
                stack.append(i)
            # stack not empty - update max , case-2.1
            else:
                max_len = max(max_len, i - stack[-1])
    
    return max_len

# java

public int longestValidParentheses(String s) {
    Stack<Integer> stack = new Stack<>();
    stack.push(-1);
    int max_len = 0;

    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            stack.push(i);
        } else {
            stack.pop();
            if (stack.isEmpty()) {
                stack.push(i);
            } else {
                max_len = Math.max(max_len, i - stack.peek());
            }
        }
    }

    return max_len;        
}

```

### 591. Tag Validator

Given a string representing a code snippet, implement a tag validator to parse the code and return whether it is valid.

A code snippet is valid if all the following rules hold:

1. The code must be wrapped in a valid closed tag. Otherwise, the code is invalid.

2. A closed tag (not necessarily valid) has exactly the following format : ``<TAG_NAME>TAG_CONTENT</TAG_NAME>``. Among them, ``<TAG_NAME>`` is the start tag, and ``</TAG_NAME>`` is the end tag. The TAG_NAME in start and end tags should be the same. A closed tag is valid if and only if the ``TAG_NAME`` and ``TAG_CONTENT`` are valid.

3. A valid ``TAG_NAME`` only contain upper-case letters, and has length in range [1,9]. Otherwise, the ``TAG_NAME`` is invalid.

4. A valid ``TAG_CONTENT`` may contain other valid closed tags, cdata and any characters (see note1) EXCEPT unmatched <, unmatched start and end tag, and unmatched or closed tags with invalid TAG_NAME. Otherwise, the ``TAG_CONTENT`` is invalid.

5. A start tag is unmatched if no end tag exists with the same ``TAG_NAME``, and vice versa. However, you also need to consider the issue of unbalanced when tags are nested.

6. A ``<`` is unmatched if you cannot find a subsequent ``>``. And when you find a ``<`` or ``</``, all the subsequent characters until the next ``>`` should be parsed as TAG_NAME (not necessarily valid).

7. The cdata has the following format : ``<![CDATA[CDATA_CONTENT]]>``. The range of CDATA_CONTENT is defined as the characters between ``<![CDATA[ and the first subsequent ]]>``.

8. ``CDATA_CONTENT`` may contain any characters. The function of cdata is to forbid the validator to parse CDATA_CONTENT, so even it has some characters that can be parsed as tag (no matter valid or invalid), you should treat it as regular characters.
 

**Example 1**:

    Input: code = "<DIV>This is the first line <![CDATA[<div>]]></DIV>"
    Output: true
    Explanation: 
    The code is wrapped in a closed tag : <DIV> and </DIV>. 
    The TAG_NAME is valid, the TAG_CONTENT consists of some characters and cdata. 
    Although CDATA_CONTENT has an unmatched start tag with invalid TAG_NAME, it should be considered as plain text, not parsed as a tag.
    So TAG_CONTENT is valid, and then the code is valid. Thus return true.

**Example 2**:

    Input: code = "<DIV>>>  ![cdata[]] <![CDATA[<div>]>]]>]]>>]</DIV>"
    Output: true
    Explanation:
    We first separate the code into : start_tag|tag_content|end_tag.
    start_tag -> "<DIV>"
    end_tag -> "</DIV>"
    tag_content could also be separated into : text1|cdata|text2.
    text1 -> ">>  ![cdata[]] "
    cdata -> "<![CDATA[<div>]>]]>", where the CDATA_CONTENT is "<div>]>"
    text2 -> "]]>>]"
    The reason why start_tag is NOT "<DIV>>>" is because of the rule 6.
    The reason why cdata is NOT "<![CDATA[<div>]>]]>]]>" is because of the rule 7.

**Example 3**:

    Input: code = "<A>  <B> </A>   </B>"
    Output: false
    Explanation: Unbalanced. If "<A>" is closed, then "<B>" must be unmatched, and vice versa.

**Constraints**:

    1 <= code.length <= 500
    code consists of English letters, digits, '<', '>', '/', '!', '[', ']', '.', and ' '.

Think of this problem like building a strict HTML/XML checker, but with a few very specific rules.

At its core, **Tag Validator** checks whether a given string is a correctly structured document enclosed inside a pair of tags.

---

**Rule 1: The Root Wrap (Outer Enclosure)**

* The **entire string** must be wrapped in a single valid start tag `<TAG_NAME>` and end tag `</TAG_NAME>`.
* Anything written *outside* this primary outer tag makes the whole string invalid.
* **Valid:** `<A>hello</A>`
* **Invalid:** `<A>hello</A><B>world</B>` (Two root elements, not wrapped in one outer tag)
* **Invalid:** `hello<A>world</A>` (Text outside the main tag)

---

**Rule 2: Tag Name Rules**

Every tag name (`<TAG_NAME>`) must follow strict limits:

1. Must contain **only UPPERCASE English letters** (`A-Z`).
2. Length must be **between 1 and 9 characters**.

* **Valid:** `<DIV>`, `<A>`, `<HELLOTHERE>`
* **Invalid:** `<div>` (lowercase), `<TAGNAMEISWAYTOOOOLONG>` (length > 9), `<A123>` (contains numbers)

---

**Rule 3: Proper Nesting (Stack Rule)**

* Tags must open and close in correct order, just like balanced parentheses `()`.
* Every open tag needs a matching close tag of the *exact same name*.
* **Valid:** `<A><B></B></A>`
* **Invalid:** `<A><B></A></B>` (Mismatched order)
* **Invalid:** `<A><B></A>` (Unclosed `<B>`)

---

**Rule 4: CDATA Blocks**

CDATA stands for "Character Data". It looks like this:

```xml
<![CDATA[CDATA_CONTENT]]>

```

* **The Golden Rule of CDATA:** Inside a CDATA block, **all rules are suspended**.
* Any `<` or `>` symbols inside CDATA are ignored and treated as ordinary plain text. You do not check for valid tags inside CDATA.
* A CDATA block starts at `<![CDATA[` and ends at the very first `]]>` encountered.
* **Important Constraint:** CDATA blocks can **only** exist *inside* an open tag. They cannot be at the root level outside the main tag.

---

**The Stack Strategy**

You scan through the string character by character from left to right using a **Stack** (a list that keeps track of currently open tags):

```
String:  "<A><![CDATA[</A>]]></A>"

```

1. **Scan `<A>`:**
* Is `A` a valid tag name (1-9 uppercase letters)? Yes.
* Push `"A"` to stack. (Stack: `["A"]`)


2. **Scan `<![CDATA[</A>]]>`:**
* Is there an open tag on the stack? Yes (`"A"` is open).
* Skip past everything until after the closing `]]>`. The `</A>` inside gets completely ignored.


3. **Scan `</A>`:**
* Is `A` at the top of our stack? Yes.
* Pop `"A"` from stack. (Stack: `[]`)


4. **End of String Check:**
* Stack is empty? Yes.
* Scanned the whole string? Yes.
* **Result: Valid!**

---

**3. Common Traps to Watch For**

1. **Empty String or Plain Text Only:** `"hello"` → Invalid (no wrapping tag).
2. **Unmatched Angles:** `<A> < </A>` → Invalid (`<` without `>` inside normal content).
3. **Closing unopened tags:** `</A>` without a preceding `<A>` → Invalid.
4. **CDATA Outside Tags:** `<![CDATA[foo]]>` without being wrapped in `<TAG>...</TAG>` → Invalid.
5. **Tags closed too early:** `<A></A>extra` → Invalid (content exists outside the root tag).


**Input Code:** `"<A><![CDATA[</A>]]></A>"`

**Length `n`:** 26

| Step | `i` | Current Pointer Context | Substring / Condition Checked | Stack State | Actions & Output |
| --- | --- | --- | --- | --- | --- |
| **1** | `0` | `code[0] = '<'` | `i > 0` condition is **False**. Matches `code.startswith('<', 0)`. | `[]` | Finds `>` at index 2 (`i=2`). Tag name = `"A"`. Length is 1 (valid), uppercase (valid). Append `"A"`. Increment `i = 2 + 1 = 3`. |
| **2** | `3` | `code[3] = '<'` | `i > 0 and not stack` is **False** (stack has `"A"`). Matches `code.startswith('<![CDATA[', 3)`. | `["A"]` | `j = 3 + 9 = 12`. Finds `]]>` at index 17 (`i=17`). Skip CDATA content completely. Increment `i = 17 + 3 = 20`. |
| **3** | `20` | `code[20] = '<'` | `i > 0 and not stack` is **False**. Matches `code.startswith('</', 20)`. | `["A"]` | `j = 20 + 2 = 22`. Finds `>` at index 23 (`i=23`). Tag name = `"A"`. Length is 1 (valid), uppercase (valid). Pops `"A"` from stack. Matches! Increment `i = 23 + 1 = 24`. |
| **4** | `24` | `code[24] = ' '` (End of String check) | Loop terminates as `i == n` (24 == 24). | `[]` | Checks `return not stack` $\rightarrow$ `not []` evaluates to **`True`**. |

**Final Output:** `True`

---

```python

def isValid(self, code: str) -> bool:
    stack = []
    i, n = 0, len(code)
    
    while i < n:
        # RULE CHECK 1: Root Tag Enclosure
        # If we are past index 0 and the stack is empty, it means the 
        # outer root tag has already closed, but there is extra code left over.
        if i > 0 and not stack:
            return False
        
        # RULE CHECK 2: CDATA Block Parsing
        if code.startswith('<![CDATA[', i):
            j = i + 9  # Move pointer past '<![CDATA['
            try:
                # Find the ending delimiter ']]>' for the CDATA block
                i = code.index(']]>', j)
            except ValueError:
                # ']]>' not found -> invalid CDATA
                return False
            # Advance pointer past ']]>'
            i += 3
        
        # RULE CHECK 3: End Tag Parsing (e.g., </TAG_NAME>)
        elif code.startswith('</', i):
            j = i + 2  # Move pointer past '</'
            try:
                # Find the closing angle bracket '>'
                i = code.index('>', j)
            except ValueError:
                # '>' not found -> invalid tag
                return False
            
            # Tag length must be between 1 and 9 characters
            if i == j or i - j > 9:
                return False
            
            # Tag name must contain ONLY uppercase letters
            for k in range(j, i):
                if not code[k].isupper():
                    return False
            
            s = code[j:i]  # Extract TAG_NAME
            
            # Must have an open tag on the stack, and it MUST match this closing tag
            if not stack or stack.pop() != s:
                return False
            
            # Advance pointer past '>'
            i += 1
        
        # RULE CHECK 4: Start Tag Parsing (e.g., <TAG_NAME>)
        elif code.startswith('<', i):
            j = i + 1  # Move pointer past '<'
            try:
                # Find the closing angle bracket '>'
                i = code.index('>', j)
            except ValueError:
                # '>' not found -> invalid tag
                return False
            
            # Tag length must be between 1 and 9 characters
            if i == j or i - j > 9:
                return False
            
            # Tag name must contain ONLY uppercase letters
            for k in range(j, i):
                if not code[k].isupper():
                    return False
            
            s = code[j:i]  # Extract TAG_NAME
            stack.append(s) # Push open tag onto stack
            
            # Advance pointer past '>'
            i += 1
        
        # RULE CHECK 5: Plain Text Characters
        else:
            # Regular characters inside a tag are valid
            i += 1
            
    # String is valid ONLY if all open tags were properly closed
    return not stack

```


## Pattern 2 — Expression Evaluation

| Section | Details |
|---|---|
| Recognition | Calculator • Expression • RPN • Postfix/Prefix |
| Core Idea | Number stack + operator stack, or a single stack depending on notation. |
| Representative Problems | 150, 224, 227, 772 |

---
