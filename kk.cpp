#include<bits/stdc++.h>
using namespace std;
bool arr[9000000001];
vector<int> vect;
void prime() {
    int max = 9000000000;
    arr[0] = arr[1] = true;
    for (int i = 2; i * i <= max; i++) {
        if (!arr[i]) {
            for (int j = i * i; j <= max; j += i) {
                arr[j] = true;
            }
        }
    }
    for (int i = 2; i <= max; i++) {
        if (!arr[i]) {
            vect.push_back(i);
        }
    }
}
int main()
{
#ifndef ONLINE_JUDGE
    freopen("INPT.txt", "r", stdin);
    freopen("OTPT.txt", "w", stdout);
    freopen("ERR.txt", "w", stderr);
#endif
    int q, k;
    cin >> q;
    prime();
    while (q--) {
        cin >> k, cout << vect[k - 1] << '\n';
    }
    return 0;
}