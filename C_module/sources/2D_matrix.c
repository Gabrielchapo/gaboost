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

void		print_t_2D_matrix_shapes(t_2D_matrix matrix)
{
	printf("shape: (%d, %d)\n", matrix.nb_row, matrix.nb_col);
}

void		free_matrix(t_2D_matrix a)
{
	if (a.values)
		free(a.values);
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

void		transpose_matrix(t_2D_matrix *a)
{
	t_2D_matrix matrix = deep_copy_matrix(*a);
	a->nb_row = matrix.nb_col;
	a->nb_col = matrix.nb_row;

	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
		a->values[i] = matrix.values[(i % matrix.nb_row) * matrix.nb_col + (i / matrix.nb_row)];
	free_matrix(matrix);
}

t_2D_matrix	create_matrix_with_val(int nb_row, int nb_col, double val)
{
	t_2D_matrix matrix;

	matrix.nb_row = nb_row;
	matrix.nb_col = nb_col;
	if (!(matrix.values = (double*)malloc(sizeof(double) * matrix.nb_col * matrix.nb_row)))
	{
		PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
		return matrix;
	}
	for (int i = 0; i < matrix.nb_row * matrix.nb_col; i++)
		matrix.values[i] = val;
	return matrix;
}

t_2D_matrix	deep_copy_matrix(t_2D_matrix a)
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
		matrix.values[i] = a.values[i];
	return matrix;
}

void		copy_matrix(t_2D_matrix a, t_2D_matrix b)
{
	// A = B
	if (a.nb_col == b.nb_col && a.nb_row == b.nb_row)
	{
		for (int i = 0; i < a.nb_row * a.nb_col; i++)
			a.values[i] = b.values[i];
	}
	else
		PyErr_SetString(PyExc_ValueError, "Value error in copy");
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
void		dot_productt(t_2D_matrix a, t_2D_matrix b, t_2D_matrix c)
{
	if (a.nb_col == b.nb_row && c.nb_row == a.nb_row && c.nb_col == b.nb_col)
	{
		for (int i = 0; i < c.nb_row * c.nb_col; i++)
		{
			c.values[i] = 0;
			for (int j = 0; j < a.nb_col; j++)
				c.values[i] += a.values[(i / c.nb_col) * a.nb_col + j] * b.values[j * b.nb_col + (i % c.nb_col)];
		}
	}
	else
		PyErr_SetString(PyExc_ValueError, "Value error in dot product");
}

void		diff_t_2D_matrix(t_2D_matrix a, t_2D_matrix b)
{
	// A = A - B
	if (a.nb_col == b.nb_col && a.nb_row == b.nb_row)
	{
		for (int i = 0; i < a.nb_row * a.nb_col; i++)
			a.values[i] -= b.values[i];
	}
	else
		PyErr_SetString(PyExc_ValueError, "Value error in difference");
}

void		mult_t_2D_matrix(t_2D_matrix a, t_2D_matrix b)
{
	// A = A * B
	if (a.nb_col == b.nb_col && a.nb_row == b.nb_row)
	{
		for (int i = 0; i < a.nb_row * a.nb_col; i++)
			a.values[i] *= b.values[i];
	}
	else
		PyErr_SetString(PyExc_ValueError, "Value error in multiplication");
}

void		scalar_mult_t_2D_matrix(t_2D_matrix a, double b)
{
	for (int i = 0; i < a.nb_row * a.nb_col; i++)
		a.values[i] *= b;
}

void		sum_in_t_2D_matrix(t_2D_matrix a, t_2D_matrix b)
{
	// A[col] = sum(B[0:n,col])
	if (a.nb_row == 1 && a.nb_col == b.nb_col)
	{
		for (int i = 0; i < b.nb_col; i++)
		{
			a.values[i] = 0;
			for (int j = 0; j < b.nb_row; j++)
				a.values[i] += b.values[j * b.nb_col + i];
		}
	}
	else
		PyErr_SetString(PyExc_ValueError, "Value error in addition");
}

void	myprint(int nb_row, int nb_col, float matrix[nb_row][nb_col])
{
	for (int i = 0; i < nb_row; i++)
	{
		for (int j = 0; j < nb_col; j++)
			printf("%f ", matrix[i][j]);
		printf("\n");
	}
}

void	dot(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ax][by])
{
	for (int i = 0; i < ax; i++)
	{
		for (int j = 0; j < by; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < ay; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	}
}
void	dot_T(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ay][by])
{
	for (int i = 0; i < ay; i++)
	{
		for (int j = 0; j < by; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < ax; k++)
				c[i][j] += a[k][i] * b[k][j];
		}
	}
}