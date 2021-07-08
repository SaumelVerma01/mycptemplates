#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define MOD 1000000007
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL); cout.tie(NULL);
#ifndef ONLINE_JUDGE
	freopen("INPT.txt", "r", stdin);
	freopen("OTPT.txt", "w", stdout);
#endif
	int n, m;
	cin >> n;
	cin >> m;
	ll a[n], b[m];
	for (int i = 0; i < n; i++) {
		cin >> a[i];
	}
	for (int i = 0; i < m; i++) {
		cin >> b[i];
	}
	sort(a, a + n);
	sort(b, b + m);
	ll ans = 0LL;
	if (n == m) {
		for (int i = 0; i < n; i++) {
			ans = (ans + abs(a[i] - b[i])) % MOD;
		}
		cout << ans << "\n";
	}
	else {
		if (n < m) {
			bool visited[m] = {false};
			for (int i = 0; i < n; i++) {
				int mini = INT_MAX;
				int ind = -1;
				for (int j = 0; j < m; j++) {
					if (abs(a[i] - b[j]) < mini && !visited[j]) {
						mini = abs(a[i] - b[j]);
						ind = j;
					}
				}
				visited[ind] = true;
				ans = (ans + abs(a[i] - b[ind])) % MOD;
			}
			cout << ans << "\n";
		}
	}
	return 0;
}