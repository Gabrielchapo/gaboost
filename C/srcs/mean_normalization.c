#include "machine_learning.h"

void	mean_normalization(t_matrix *matrix, int column)
{
	double average = 0;
	double min;
	double max;
	double range;

	if (matrix->value[column])
	{
		min = matrix->value[column];
		max = matrix->value[column];
	}
	for (int j = 0 ; j < matrix->nb_row ; j++)
	{
		if (matrix->value[matrix->nb_column * j + column] < min)
			min = matrix->value[matrix->nb_column * j + column];
		if (matrix->value[matrix->nb_column * j + column] > max)
			max = matrix->value[matrix->nb_column * j + column];
		average += matrix->value[matrix->nb_column * j + column];
	}
	range = max - min;
	average /= (double)matrix->nb_row;
	for (int j = 0 ; j < matrix->nb_row ; j++)
		matrix->value[matrix->nb_column * j + column] =
			(matrix->value[matrix->nb_column * j + column] - average) / range;
	printf("range %f average %f\n", range, average);
}
