#include "machine_learning.h"

double	linear_function_mutliple(t_matrix *matrix_a, double *thetas, int i)
{
	double result;

	result = thetas[0];
	for (int k = 1 ; k < matrix_a->nb_column + 1 ; k++)
		result += thetas[k] * matrix_a->value[i * matrix_a->nb_column + k];
	return (result);
}

double	derivated_cost_theta(t_matrix *matrix_a, t_matrix *matrix_b, double *thetas, int j)
{
	double result = 0;
	double tmp;
	
	for (int i = 0 ; i < matrix_a->nb_row ; i++)
	{
		tmp = linear_function_mutliple(matrix_a, thetas, i);
		if (j != 0)
			result += (tmp - matrix_b->value[i]) * matrix_a->value[matrix_a->nb_column * i + j - 1];
		else
			result += (tmp - matrix_b->value[i]);
	}
	result /= (double)matrix_a->nb_row;
	return (result);
}

t_matrix	*linear_gradientDescent(t_matrix *matrix_a, t_matrix *matrix_b, double gradient, int nb_iterations)
{
	double thetas[matrix_a->nb_column + 1];
	double tmp[matrix_a->nb_column + 1];

	for (int i = 0 ; i < nb_iterations ; i++)
	{
		for (int j = 0 ; j < matrix_a->nb_column + 1; j++)
			tmp[j] = gradient * derivated_cost_theta(matrix_a, matrix_b, thetas, j);
		for (int j = 0 ; j < matrix_a->nb_column + 1; j++)
			thetas[j] -= tmp[j];
	}
	return (create_matrix(thetas, 1, matrix_a->nb_column + 1));
}