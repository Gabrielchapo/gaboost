#include "../include/binding.h"

t_2D_matrix	initialize_weights(int nb_row, int nb_col)
{
	t_2D_matrix matrix;

	matrix.nb_row = nb_row;
	matrix.nb_col = nb_col;
	if (!(matrix.values = (double*)malloc(sizeof(double) * matrix.nb_col * matrix.nb_row)))
	{
		PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
		return matrix;
	}
	for (int i = 0; i < matrix.nb_col * matrix.nb_row; i++)
		matrix.values[i] = 1;
	return matrix;
}
