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

void		sigmoid(t_2D_matrix matrix)
{
	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
	{
		matrix.values[i] = 1 / (1 + exp(-matrix.values[i]));
	}
}

t_2D_matrix	sigmoid_derv(t_2D_matrix a)
{
	t_2D_matrix matrix;

	matrix.nb_row = a.nb_row;
	matrix.nb_col = a.nb_col;
	if (!(matrix.values = (double*)malloc(sizeof(double) * matrix.nb_col * matrix.nb_row)))
	{
		PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
		return matrix;
	}
	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
		matrix.values[i] = a.values[i] * (1.0 - a.values[i]);
	return matrix;
}

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

double		cross_entropy(t_2D_matrix X, t_2D_matrix Y, t_2D_matrix W)
{
	double loss = 0;
	t_2D_matrix predicted = dot_product(X, W);
	softmax(predicted);
	for (int i = 0; i < predicted.nb_row * predicted.nb_col; i++)
		predicted.values[i] = Y.values[i] * log(predicted.values[i]) + ((1 - Y.values[i]) * log(1.0 - predicted.values[i]));
	double sum[predicted.nb_row];
	for (int i = 0; i < predicted.nb_row; i++)
	{
		sum[i] = 0;
		for (int j = 0 ; j < predicted.nb_col ; j++)
			sum[i] += predicted.values[i * predicted.nb_col + j];
		loss += sum[i];
	}
	loss /= predicted.nb_row;
	loss *= -1;
	free(predicted.values);
	return loss;
}