#include "../include/binding.h"

double		mse(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ay][by])
{
	double mse = 0;
	double tmp = 0;

	for (int i = 0; i < ax; i++)
	{
		tmp = 0;
		for (int j = 0; j < ay; j++)
			tmp += a[i][j] * c[j][0];
		tmp -= b[i][0]; 
		mse += pow(tmp, 2);
	}
	mse /= ax;
	return mse;
}

double		ce(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ay][by], float d[ax][by])
{
	double loss = 0;
	dot(ax, ay, a, ay, by, c, d);
	sm(ax, by, d);
	for (int i = 0; i < ax; i++) {
		for (int j = 0 ; j < by ; j++)
			d[i][j] = b[i][j] * log(d[i][j]) + ((1.0 - b[i][j]) * log(1.0 - d[i][j]));
	}
	double sum[ax];
	for (int i = 0; i < ax; i++) {
		sum[i] = 0;
		for (int j = 0; j < by; j++)
			sum[i] += d[i][j];
		loss += sum[i];
	}
	loss /= ax;
	loss *= -1;
	return loss;
}