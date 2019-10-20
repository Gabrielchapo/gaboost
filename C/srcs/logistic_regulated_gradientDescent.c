#include "machine_learning.h"

double		hypothesis(t_matrix *X, double *thetas, int line)
{
	double result = 0;
	double tmp = 0;

	for (int k = 0 ; k < X->nb_column + 1 ; k++)
		tmp += thetas[k] * X->value[line * X->nb_column + k];
	tmp *= -1;
	result = 1.0 / (1.0 + exp(tmp));
	return (result);
}

double	derivated_cost(t_matrix *X, t_matrix *Y, t_matrix *lambda, double *thetas, int j)
{
	double result = 0;
	double tmp;
	
	for (int i = 0 ; i < X->nb_row ; i++)
	{
		tmp = hypothesis(X, thetas, i) - Y->value[i];
		if (j != 0)
			result += tmp * X->value[X->nb_column * i + j - 1] + (lambda->value[j - 1] / (double)X->nb_row * thetas[j]);
		else
			result += tmp;
	}
	printf("result : %f\n", result);
	return (result);
}

t_matrix	*logistic_regulated_gradientDescent(t_matrix *X, t_matrix *Y, t_matrix *lambda, double gradient, int nb_iterations)
{
	int nb_thetas = X->nb_column + 1;
	double thetas[nb_thetas];
	double tmp[nb_thetas];

	for (int i = 0 ; i < nb_iterations ; i++)
	{
		printf("theta0 : %f | theta1 : %f | theta2 : %f \n", thetas[0],thetas[1],thetas[2]);
		for (int j = 0 ; j < nb_thetas; j++)
			tmp[j] = (gradient / (double)X->nb_row) * derivated_cost(X, Y, lambda, thetas, j);
		for (int j = 0 ; j < nb_thetas; j++)
			thetas[j] -= tmp[j];
	}
	return (create_matrix(thetas, 1, nb_thetas));
}
