#include "../include/binding.h"

void		print_t_2D_matrix(t_2D_matrix matrix)
{
	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
	{
		if (i != 0 && i % matrix.nb_col == 0)
			printf("\n");
		printf("%f ", matrix.values[i]);
	}
	printf("\n");
}

t_2D_matrix	transposed_matrix(t_2D_matrix a)
{
	t_2D_matrix matrix;

	matrix.nb_row = a.nb_col;
	matrix.nb_col = a.nb_row;
	if (!(matrix.values = (double*)malloc(sizeof(double) * matrix.nb_col * matrix.nb_row)))
	{
		PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
		return matrix;
	}
	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
		matrix.values[i] = a.values[(i % matrix.nb_col) * a.nb_col + (i / matrix.nb_col)];
	return matrix;
}

t_2D_matrix	dot_product(t_2D_matrix a, t_2D_matrix b)
{
	t_2D_matrix matrix;

	matrix.nb_row = a.nb_row;
	matrix.nb_col = b.nb_col;
	if (a.nb_col == b.nb_row)
	{
		if (!(matrix.values = (double*)malloc(sizeof(double) * matrix.nb_col * matrix.nb_row)))
		{
			PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
			return matrix;
		}
		for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
		{
			matrix.values[i] = 0;
			for (int j = 0; j < a.nb_col; j++)
				matrix.values[i] += a.values[(i / matrix.nb_col) * a.nb_col + j] * b.values[j * b.nb_col + (i % matrix.nb_col)];
		}
		return matrix;
	}
	else
	{
		PyErr_SetString(PyExc_ValueError, "Value error in dot product");
		return matrix;
	}
}
