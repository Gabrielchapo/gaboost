#include "machine_learning.h"

double		get_value(t_matrix *matrix, int column, int row)
{
	return (matrix->value[matrix->nb_column * row + column]);
}

void		print_matrix(t_matrix *matrix)
{
	int i;
	int j;

	i = -1;
	while (++i < matrix->nb_row)
	{
		j = -1;
		while (++j < matrix->nb_column)
			printf("%f ", matrix->value[matrix->nb_column * i + j]);
		printf("\n");
	}
	printf("\n");
}

t_matrix	*create_matrix(double *value, int nb_column, int nb_row)
{
	t_matrix	*matrix;
	int			i;

	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return(NULL);
	if (!(matrix->value = ft_memalloc(sizeof(double) * nb_column * nb_row)))
		return(NULL);
	matrix->nb_column = nb_column;
	matrix->nb_row = nb_row;
	i = -1;
	while (++i < nb_row * nb_column)
		matrix->value[i] = (value == NULL) ? 0 : value[i];
	return (matrix);
}

void		free_matrix(t_matrix *matrix)
{
	if (matrix)
	{
		if (matrix->value)
			ft_memdel((void**)&(matrix->value));
        ft_bzero(matrix, sizeof(t_matrix));
        ft_memdel((void**)&matrix);
	}
}

t_matrix	*add_matrixes(t_matrix *matrix_a, t_matrix *matrix_b)
{
	t_matrix 	*matrix;
	int 		i;

	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return (NULL);
	if (matrix_a->nb_column != matrix_b->nb_column || matrix_a->nb_row != matrix_b->nb_row)
		return (NULL);
	if (!(matrix->value = ft_memalloc(sizeof(double) * matrix_a->nb_column * matrix_a->nb_row)))
		return(NULL);
	matrix->nb_column = matrix_a->nb_column;
	matrix->nb_row = matrix_a->nb_row;
	i = -1;
	while (++i < matrix->nb_row * matrix->nb_column)
		matrix->value[i] = matrix_a->value[i] + matrix_b->value[i];
	return (matrix);
}

t_matrix	*multiply_matrixes(t_matrix *matrix_a, t_matrix *matrix_b)
{
	t_matrix 	*matrix;

	if (matrix_a->nb_column != matrix_b->nb_row)
		return (NULL);
	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return (NULL);
	matrix->nb_column = matrix_b->nb_column;
	matrix->nb_row = matrix_a->nb_row;
	if (!(matrix->value = ft_memalloc(sizeof(double) * matrix->nb_row * matrix->nb_column)))
		return(NULL);
	for (int i = 0 ; i < matrix->nb_row * matrix->nb_column ; i++)
	{
		matrix->value[i] = 0;
		for (int k = 0 ; k < matrix->nb_row ; k++)
		{
			matrix->value[i] += matrix_a->value[i / matrix->nb_column * matrix_a->nb_column + k]
				* matrix_b->value[k * matrix_b->nb_column + i % matrix_b->nb_column];
		}
	}
	return (matrix);
}

t_matrix	*sub_matrixes(t_matrix *matrix_a, t_matrix *matrix_b)
{
	t_matrix 	*matrix;
	int 		i;

	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return (NULL);
	if (matrix_a->nb_column != matrix_b->nb_column || matrix_a->nb_row != matrix_b->nb_row)
		return (NULL);
	if (!(matrix->value = ft_memalloc(sizeof(double) * matrix_a->nb_column * matrix_a->nb_row)))
		return(NULL);
	matrix->nb_column = matrix_a->nb_column;
	matrix->nb_row = matrix_a->nb_row;
	i = -1;
	while (++i < matrix->nb_row * matrix->nb_column)
		matrix->value[i] = matrix_a->value[i] - matrix_b->value[i];
	return (matrix);
}

t_matrix	*scalar_mult(t_matrix *matrix_a, double scalar)
{
	t_matrix 	*matrix;
	int 		i;

	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return (NULL);
	if (!(matrix->value = ft_memalloc(sizeof(double) * matrix_a->nb_column * matrix_a->nb_row)))
		return(NULL);
	matrix->nb_column = matrix_a->nb_column;
	matrix->nb_row = matrix_a->nb_row;
	i = -1;
	while (++i < matrix->nb_row * matrix->nb_column)
		matrix->value[i] = matrix_a->value[i] * scalar;
	return (matrix);
}

t_matrix	*scalar_div(t_matrix *matrix_a, double scalar)
{
	t_matrix 	*matrix;
	int 		i;

	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return (NULL);
	if (scalar == 0.0)
		return (NULL);
	if (!(matrix->value = ft_memalloc(sizeof(double) * matrix_a->nb_column * matrix_a->nb_row)))
		return(NULL);
	matrix->nb_column = matrix_a->nb_column;
	matrix->nb_row = matrix_a->nb_row;
	i = -1;
	while (++i < matrix->nb_row * matrix->nb_column)
		matrix->value[i] = matrix_a->value[i] / scalar;
	return (matrix);
}

t_matrix	*transpose_matrix(t_matrix *matrix_a)
{
	t_matrix 	*matrix;
	int 		i;
	int			j;

	if (!(matrix = ft_memalloc(sizeof(t_matrix))))
		return (NULL);
	if (!(matrix->value = ft_memalloc(sizeof(double) * matrix_a->nb_column * matrix_a->nb_row)))
		return(NULL);
	matrix->nb_column = matrix_a->nb_row;
	matrix->nb_row = matrix_a->nb_column;
	i = -1;
	while (++i < matrix_a->nb_column)
	{
		j = -1;
		while (++j < matrix_a->nb_row)
			matrix->value[i * matrix->nb_column + j] = matrix_a->value[j * matrix_a->nb_column + i];
	}
	return (matrix);
}

double		get_determinant(double *value, int n)
{
	int		sign = 1;
	int		k;
	double	result = 0;
	double	tmp[(n - 1) * (n - 1)];

	if (n == 1)
		return (value[0]);
	for (int f = 0; f < n; f++)
    {
		k = 0;
		for (int i = 1; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (f != j)
					tmp[k++] = value[n * i + j];
			}
		}
		result += sign * value[f] * get_determinant(tmp, n - 1); 
		sign = -sign; 
    }
	return (result);
}

t_matrix	*adj_matrix(t_matrix *matrix_a)
{
	int		size_tmp = (matrix_a->nb_column - 1) * (matrix_a->nb_row - 1);
	double	tmp[size_tmp];
	int		k;
	t_matrix *matrix;
	int		sign = 1;
	
	if (!(matrix = create_matrix(NULL, matrix_a->nb_row, matrix_a->nb_column)))
		return (NULL);
	for (int l = 0; l < matrix_a->nb_column * matrix_a->nb_row; l++)
	{
		k = 0;
		for (int i = 0; i < matrix_a->nb_column * matrix_a->nb_row; i++)
		{
			if (i % matrix_a->nb_column != l % matrix_a->nb_column && i / matrix_a->nb_row != l / matrix_a->nb_row)
				tmp[k++] = matrix_a->value[i];
		}
		sign = ((l / matrix_a->nb_row + l % matrix_a->nb_row) % 2 == 0) ? 1: -1; 
		matrix->value[l] = sign * get_determinant(tmp, matrix_a->nb_column - 1);
	}
	return (matrix);
}

t_matrix	*inverse_matrix(t_matrix *matrix_a)
{
	t_matrix 	*matrix;
	t_matrix	*matrix_b;
	t_matrix	*matrix_c;
	double		determinant;

	if (matrix_a->nb_column != matrix_a->nb_row)
		return (NULL);
	determinant = get_determinant(matrix_a->value, matrix_a->nb_column);
	if (determinant == 0)
		return (NULL);
	if (!(matrix = transpose_matrix(matrix_a)))
		return (NULL);
	if (!(matrix_b = adj_matrix(matrix)))
		return (NULL);
	if (!(matrix_c = scalar_div(matrix_b, determinant)))
		return (NULL);
	free_matrix(matrix);
	free_matrix(matrix_b);
	return (matrix_c);
}