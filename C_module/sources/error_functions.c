#include "../include/binding.h"

double		mean_square_error(t_2D_matrix X, t_2D_matrix Y, t_2D_matrix W)
{
	double mse = 0;
	double tmp = 0;

	for (int i = 0; i < X.nb_row; i++)
	{
		tmp = 0;
		for (int j = 0; j < X.nb_col; j++)
			tmp += X.values[i * X.nb_col + j] * W.values[j];
		tmp -= Y.values[i]; 
		mse += pow(tmp, 2);
	}
	mse /= X.nb_row;
	return mse;
}
