#include "machine_learning.h"

t_matrix	*linear_normalEquation(t_matrix *X, t_matrix *Y)
{
	t_matrix *matrix_a;
	t_matrix *matrix_b;
	t_matrix *matrix_c;
	t_matrix *matrix_d;
	t_matrix *matrix_e;

	if (!(matrix_a = transpose_matrix(X)))
		return (NULL);
	
	if (!(matrix_b = multiply_matrixes(matrix_a, X)))
		return (NULL);
	
	if (!(matrix_c = inverse_matrix(matrix_b)))
		return (NULL);
	
	if (!(matrix_d = multiply_matrixes(matrix_c, matrix_a)))
		return (NULL);
	
	if (!(matrix_e = multiply_matrixes(matrix_d, Y)))
		return (NULL);
	
	free_matrix(matrix_a);
	free_matrix(matrix_b);
	free_matrix(matrix_c);
	free_matrix(matrix_d);
	return (matrix_e);
}