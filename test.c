#include <stdio.h>
#include <stdlib.h>

void	dot(float a[100][100], float b[100][100], float c[100][100])
{
	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < 100; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	}
}

void		dot_productt(float *a, float *b, float *c)
{
    for (int i = 0; i < 10000; i++)
    {
        c[i] = 0;
        for (int j = 0; j < 100; j++)
            c[i] += a[(i / 100) * 100 + j] * b[j * 100 + (i % 100)];
    }
}

int main()
{
    float a[100][100];
    for (int i=0; i<100;i++)
    {
        for (int j=0;j<100;j++)
            a[i][j] = i;
    }
    float b[100][100];
    for (int i=0; i<100;i++)
    {
        for (int j=0;j<100;j++)
            b[i][j] = j;
    }
    float c[100][100];
    for (int i=0; i<100;i++)
    {
        for (int j=0;j<100;j++)
            c[i][j] = i;
    }/*
    float *a;
    float *b;
    float *c;
    a = (float*)malloc(sizeof(float)*ax);
    b = (float*)malloc(sizeof(float)*ax);
    c = (float*)malloc(sizeof(float)*ax);
    for (int i=0; i < 10000; i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }
    */
    for (int i = 0; i < 10000;i++)
    {
        //dot_productt(a,b,c);
        dot(a,b,c);
    }
        

    return 0;
}