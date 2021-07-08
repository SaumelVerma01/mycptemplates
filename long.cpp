#include <bits/stdc++.h>
using namespace std;
main()
{
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    freopen("error.txt", "w", stderr);
#endif
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int T;
    cin >> T;
    while (T--)
    {
        int n, k;
        cin >> n >> k;
        int Y = 0;
        for (int y = 0; y < n; y++)
        {
            if (y == 0)
            {
                cout << y << endl;
                int x;
                cin >> x;
                if (x == 1)
                {
                    break;
                }
            }
            else
            {
                cout << (y ^ (y - 1)) << endl;
                int x;
                cin >> x;
                if (x == 1)
                {
                    break;
                }
            }
        }
    }
}