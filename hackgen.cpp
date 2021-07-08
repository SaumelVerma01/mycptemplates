#include <bits/stdc++.h>
using namespace std;
int main()
{
	cout << 1 << "\n";
	cout << 200000 << "\n";
	for (int i = 1; i <= 100000; i++)
	{
		cout << 200000 << " ";
	}
	for (int i = 1; i <= 10; i++)
	{
		if (i == 10)
		{
			for (int j = 1; j <= 10000; j++)
			{
				if (j == 10000)
				{
					cout << i << "\n";
				}
				else
				{
					cout << i << " ";
				}
			}
			continue;
		}
		for (int j = 1; j <= 10000; j++)
		{
			cout << i << " ";
		}
	}
	for (int i = 1; i <= 200000; i++)
	{
		if (i != 200000)
			cout << 1000000000 << " ";
		else
			cout << 1000000000 << "\n";
	}
}