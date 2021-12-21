#include "bits/stdc++.h"
/**********************DEFINES*********************************/
#define int int64_t
#define nline "\n"
#define all(v) v.begin(), v.end()
/*************************************************************/
const int MOD = 1e9 + 7;
const int MD = 998244353;
using namespace std;

/*************************************************************/
inline int add(int a, int b);
inline int mul(int a, int b);
template <typename T>
inline istream &operator>>(istream& in, vector<T>& a);
template <typename T>
inline ostream &operator<<(ostream& os, const vector<T>& a);
template<typename T, typename S>
inline istream &operator>>(istream& in, pair<T, S>& a);
template<typename T, typename S>
inline ostream &operator<<(ostream& os, pair<T, S>& a);
/*************************************************************/

void run_case(int tt) {
    return;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    int test_cases = 1;
    cin >> test_cases;
    for (int t = 1; t <= test_cases; t++) {
        run_case(t);
    }
}

/********************************I/O****************************************/
template <typename T>
inline istream &operator>>(istream& in, vector<T>& a) {
    for (auto &x : a)in >> x;
    return in;
}
template <typename T>
inline ostream &operator<<(ostream& os, const vector<T>& a) {
    for (auto &x : a)os << x << " ";
    os << nline;
    return os;
}
template<typename T, typename S>
inline istream &operator>>(istream& in, pair<T, S>& a) {
    in >> a.first >> a.second;
    return in;
}
template<typename T, typename S>
inline ostream &operator<<(ostream& os, pair<T, S>& a) {
    os << a.first << " " << a.second << nline;
    return os;
}
int add(int a, int b, int M) {
    return (a + b + M) % M;
}
int mul(int x, int y, int M) {
    return (1LL * x * y) % M;
}
/*************************************************************************/

/*
    Sigma rule:
    #) Always read all problems of the problemset
    #) Leave A if not done in 10 mins. :(
    1) never mix 0 based and 1 based indexing
    2) never use ceil function
    3) never use unordered_map||unordered_set
    4) always go with the gut feeling
    5) first sighted observations always lead to AC, not overthinking
    6) always take MOD, always :(
*/