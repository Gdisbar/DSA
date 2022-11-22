737. Sentence Similarity II
================================
// Given two sentences words1, words2 (each represented as an array of strings), 
// and a list of similar word pairs pairs, determine if two sentences are similar.

// For example, words1 = ["great", "acting", "skills"] and words2 = ["fine", 
// "drama", "talent"] are similar, if the similar word pairs are 
// pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], 
// 	["skills","talent"]].

// Note that the similarity relation is transitive. For example, if “great” and 
// “good” are similar, and “fine” and “good” are similar, then “great” and “fine” 
// are similar.

// Similarity is also symmetric. For example, “great” and “fine” being similar is 
// the same as “fine” and “great” being similar.

// Also, a word is always similar with itself. For example, the sentences 
// words1 = ["great"], words2 = ["great"], pairs = [] are similar, even though 
// there are no specified similar word pairs.

// Finally, sentences can only be similar if they have the same number of words. 
// So a sentence like words1 = ["great"] can never be similar to 
// words2 = ["doubleplus","good"].

// Note:

//     The length of words1 and words2 will not exceed 1000.
//     The length of pairs will not exceed 2000.
//     The length of each pairs[i] will be 2.
//     The length of each words[i] and pairs[i][j] will be in the range [1, 20].

// This is more complicated than Sentence Similarity I. To check if two words 
// are similar given the transitivity (for example If “A” = “B”, “B” = “C”, then 
// “A” = “C”), we can use a graph to help us connect all similar words together. 
// Then for each word pairs, we start from the source word, using DFS to find the 
// destination word. In case of we do DFS to the same node twice, we can create a 
// set to keep a record of visited nodes.

// Let’s walk through a simple example:

// words1 = [“A”, “B”, “C”], words2 = [“D”, “E”, “F”]

// pairs[][] = [“A”, “G”],[“D”, “G”],[“B”, “E”],[“C”, “F”]

// We construct the graph using a map to represent it.
// String |set<String>
// A | [G]
// G | [A, D]
// D | [G]
// B | [E]
// E | [B]
// C | [F]
// F | [C]

// Now we begin to check A and D. A is the source, D is the target.

// We go to the entry where the key is A, and check if this set contains D. 
// It doesn’t, but contains G! It’s possible G has a set contains D, so we 
// change our source from A to G and keep finding out. We get the set [A, D], 
// and we need to check each word here. The first one is A again, oh we just 
// checked this! We don’t want to go to an endless loop. So we need to skip this, 
// and we see D. It’s equal to the target! We are done. Well, on the opposite, 
// if we are not this lucky, we need to keep finding. After we go through the 
// entire map we still can’t find the target, we failed.



// Time complexity: O(|Pairs| + |words1|)
// Space complexity: O(|Pairs|)

class UnionFindSet{
	unordered_map<string, string> parents_;

	 bool Union(const string& word1, const string& word2) {
	        const string& p1 = Find(word1, true);
	        const string& p2 = Find(word2, true);
	        if (p1 == p2) return false;        
	        parents_[p1] = p2;
	        return true;
	 }


	const string& Find(const string& word, bool create = false) {
	        if (!parents_.count(word)) {
	            if (!create) return word;
	            return parents_[word] = word;
	        }
	        string w = word;
	        while (w != parents_[w]) {
	            parents_[w] = parents_[parents_[w]];
	            w = parents_[w];
	        }
	        return parents_[w];
	    }
};

class Solution {
public:

    bool areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, 
    	vector<pair<string, string>>& pairs) {
        if (words1.size() != words2.size()) return false;
        UnionFindSet s;
        for (const auto& pair : pairs)
            s.Union(pair.first, pair.second);
        for (int i = 0; i < words1.size(); ++i) 
            if (s.Find(words1[i]) != s.Find(words2[i])) return false;
        return true;
    }


/// Using DFS
/// Time Complexity: O(len(words)^2)
/// Space Complexity: O(len(words)^2)

class Solution{
private:
	unordered_map<string, int> ids_;
	unordered_map<string, unordered_set<string>> g_;
	bool dfs(const string& curr, int id) {
	        ids_[curr] = id;        
	        for (const auto& next : g_[curr]) {
	            if (ids_.count(next)) continue;
	            if (dfs(next, id)) return true;
	        }
	        return false;
	}
public:
	bool areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, 
						vector<pair<string, string>>& pairs) {

	        if (words1.size() != words2.size()) return false;
			g_.clear();
	        ids_.clear();
	        for (const auto& p : pairs) {
	            g_[p.first].insert(p.second);
	            g_[p.second].insert(p.first);
	        }        
	        int id = 0;
	        for (const auto& p : pairs) {
	            if(!ids_.count(p.first)) dfs(p.first, ++id);
	            if(!ids_.count(p.second)) dfs(p.second, ++id);
	        }
	        for (int i = 0; i < words1.size(); ++i) {
	            if (words1[i] == words2[i]) continue;
	            auto it1 = ids_.find(words1[i]);
	            auto it2 = ids_.find(words2[i]);
	            if (it1 == ids_.end() || it2 == ids_.end()) return false;
	            if (it1->second != it2->second) return false;
	        }
	        return true;
	    }
};



// class UnionFind{

// private:
//     int* rank;
//     int* parent; 
//     int count;   

// public:

//     UnionFind(int count){
//         parent = new int[count];
//         rank = new int[count];
//         this->count = count;
//         for( int i = 0 ; i < count ; i ++ ){
//             parent[i] = i;
//             rank[i] = 1;
//         }
//     }


//     ~UnionFind(){
//         delete[] parent;
//         delete[] rank;
//     }

//     int find(int p){
//         assert( p >= 0 && p < count );

//         // path compression 1
//         while( p != parent[p] ){
//             parent[p] = parent[parent[p]];
//             p = parent[p];
//         }
//         return p;
//     }

//     bool isConnected( int p , int q ){
//         return find(p) == find(q);
//     }

//     void unionElements(int p, int q){

//         int pRoot = find(p);
//         int qRoot = find(q);

//         if( pRoot == qRoot )
//             return;

//         if( rank[pRoot] < rank[qRoot] ){
//             parent[pRoot] = qRoot;
//         }
//         else if( rank[qRoot] < rank[pRoot]){
//             parent[qRoot] = pRoot;
//         }
//         else{ // rank[pRoot] == rank[qRoot]
//             parent[pRoot] = qRoot;
//             rank[qRoot] += 1;   // 此时, 我维护rank的值
//         }
//     }
// };


// class Solution {

// public:
//     bool areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2,
//                                 vector<pair<string, string>> pairs) {

//         if(words1.size() != words2.size())
//             return false;

//         if(words1.size() == 0)
//             return true;

//         unordered_map<string, int> wordIndex = createWordIndex(pairs);
//         UnionFind uf = createUF(wordIndex, pairs);

//         for(int i = 0 ; i < words1.size() ; i ++){
//             if(words1[i] == words2[i])
//                 continue;

//             if(wordIndex.find(words1[i]) == wordIndex.end() ||
//                wordIndex.find(words2[i]) == wordIndex.end())
//                 return false;

//             if(!uf.isConnected(wordIndex[words1[i]], wordIndex[words2[i]]))
//                 return false;
//         }

//         return true;
//     }

// private:

//     UnionFind createUF(
//             const unordered_map<string, int>& wordIndex,
//             const vector<pair<string, string>>& pairs){

//         UnionFind uf(wordIndex.size());
//         for(pair<string, string> p: pairs){
//             int i1 = wordIndex.at(p.first);
//             int i2 = wordIndex.at(p.second);
//             uf.unionElements(i1, i2);
//         }

//         return uf;
//     }

//     unordered_map<string, int> createWordIndex(const vector<pair<string, string>>& pairs){

//         unordered_map<string, int> wordIndex;
//         int index = 0;
//         for(pair<string, string> p: pairs){
//             if(wordIndex.find(p.first) == wordIndex.end())
//                 wordIndex.insert(make_pair(p.first, index ++));
//             if(wordIndex.find(p.second) == wordIndex.end())
//                 wordIndex.insert(make_pair(p.second, index ++));
//         }
//         return wordIndex;
//     }
// };


// void printBool(bool res){
//     cout << (res ? "True" : "False") << endl;
// }