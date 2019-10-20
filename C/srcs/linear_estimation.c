#include "machine_learning.h"

double	linear_estimation(t_matrix *X, int line, t_matrix *thetas)
{
	double value = 0;

	if (thetas->value[0])
		value = thetas->value[0];
	for (int j = 1 ; j < thetas->nb_row ; j++)
		value += thetas->value[j] * X->value[line * X->nb_column + j - 1];
	return (value);
}
