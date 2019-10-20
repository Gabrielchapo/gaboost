#include "machine_learning.h"

double	linear_costFunction(t_matrix *matrix_a, t_matrix *matrix_b, t_matrix *thetas)
{
	double cost = 0;
	double tmp;

	for (int i = 0 ; i < matrix_a->nb_row ; i++)
	{
		tmp = thetas->value[0];
		for (int j = 1 ; j < thetas->nb_row ; j++)
			tmp += thetas->value[j] * matrix_a->value[i * matrix_a->nb_column + j];
		tmp -= matrix_b->value[i];
		cost += pow(tmp, 2);
	}
	cost /= 2.0 * (double)matrix_a->nb_row;
	return (cost);
}
