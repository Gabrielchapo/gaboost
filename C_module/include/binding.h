#ifndef __BINDING_H__
# define __BINDING_H__

# include <Python.h>

/*
** Structure for 2D matrix
*/
typedef struct  s_2D_matrix
{
    // type only used for layers
    uint8_t     type;
    int         nb_row;
    int         nb_col;
    double      *values;
}               t_2D_matrix;

/*
** linear_regression.c
*/
PyObject    *linear_regression_fit(PyObject *self, PyObject *args);
PyObject    *linear_regression_predict(PyObject *self, PyObject *args);

/*
** logistic_regression.c
*/
PyObject    *logistic_regression_fit(PyObject *self, PyObject *args);
PyObject    *logistic_regression_predict(PyObject *self, PyObject *args);

/*
** neural_network.c
*/
PyObject    *neural_network_fit(PyObject *self, PyObject *args);
PyObject	*neural_network_predict(PyObject *self, PyObject *args);

/*
** 2D_matrix.c
*/
void        print_t_2D_matrix(t_2D_matrix matrix);
void		print_t_2D_matrix_shapes(t_2D_matrix matrix);
t_2D_matrix	create_matrix_with_val(int nb_row, int nb_col, double val);
t_2D_matrix	deep_copy_matrix(t_2D_matrix a);
t_2D_matrix	transposed_matrix(t_2D_matrix a);
void		transpose_matrix(t_2D_matrix *a);
t_2D_matrix	dot_product(t_2D_matrix a, t_2D_matrix b);
void		dot_productt(t_2D_matrix a, t_2D_matrix b, t_2D_matrix c);
void    	diff_t_2D_matrix(t_2D_matrix a, t_2D_matrix b);
void    	mult_t_2D_matrix(t_2D_matrix a, t_2D_matrix b);
void		sum_in_t_2D_matrix(t_2D_matrix a, t_2D_matrix b);
void        scalar_mult_t_2D_matrix(t_2D_matrix a, double b);
void		free_matrix(t_2D_matrix a);
void		copy_matrix(t_2D_matrix a, t_2D_matrix b);

/*
** python_utils.c
*/
PyObject	*create_PyObject_from_t_2D_matrix(t_2D_matrix a);
t_2D_matrix parsing(PyObject *list, int bias);

/*
** error_functions.c
*/
double		mean_square_error(t_2D_matrix X, t_2D_matrix Y, t_2D_matrix W);
double		cross_entropy(t_2D_matrix X, t_2D_matrix Y, t_2D_matrix W);
void		softmax(t_2D_matrix predicted);
void		sigmoid(t_2D_matrix predicted);
void    	sigmoid_derv(t_2D_matrix a);

#endif