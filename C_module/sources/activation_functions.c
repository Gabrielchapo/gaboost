#include "../include/binding.h"

void		softmax(t_2D_matrix matrix)
{
	double the_sum;

	for (int i = 0; i < matrix.nb_row; i++)
	{
		the_sum = 0;
		for (int j = 0 ; j < matrix.nb_col ; j++)
			the_sum += exp(matrix.values[i * matrix.nb_col + j]);
		the_sum = (the_sum == 0) ? 1 : the_sum;
		for (int j = 0 ; j < matrix.nb_col ; j++)
			matrix.values[i * matrix.nb_col + j] = exp(matrix.values[i * matrix.nb_col + j]) / the_sum;
	}
}

void		sm(int nb_row, int nb_col, float matrix[nb_row][nb_col])
{
	double the_sum;

	for (int i = 0; i < nb_row; i++)
	{
		the_sum = 0;
		for (int j = 0 ; j < nb_col ; j++)
			the_sum += exp(matrix[i][j]);
		the_sum = (the_sum == 0) ? 1 : the_sum;
		for (int j = 0 ; j < nb_col ; j++)
			matrix[i][j] = exp(matrix[i][j]) / the_sum;
	}
}

void		sigmoid(t_2D_matrix matrix)
{
	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
		matrix.values[i] = 1.0 / (1.0 + exp(-matrix.values[i]));
}

void		sigmoid_derv(t_2D_matrix a)
{
	for (int i = 0; i < a.nb_row * a.nb_col; i++)
		a.values[i] *= (1.0 - a.values[i]);
}
