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

typedef struct  s_size
{
    int         nb_row;
    int         nb_col;
}               t_size;

PyObject	*regression_fit(PyObject *self, PyObject *args);
PyObject	*regression_predict(PyObject *self, PyObject *args);
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

void	myprint(int nb_row, int nb_col, float matrix[nb_row][nb_col]);
void	dot(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ax][by]);
void	dot_T(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ay][by]);


/*
** python_utils.c
*/
PyObject	*create_PyObject_from_t_2D_matrix(t_2D_matrix a);
t_2D_matrix parsing(PyObject *list, int bias);
void	    parsingg(PyObject *list, int bias, int nb_row, int nb_col, float matrix[nb_row][nb_col]);
PyObject	*create_return(int nb_row, int nb_col, float matrix[nb_row][nb_col]);

/*
** error_functions.c
*/
double		mse(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ay][by]);
double		ce(int ax, int ay, float a[ax][ay], int bx, int by, float b[bx][by], float c[ay][by], float d[ax][by]);

/*
** activation_functions.c
*/
void		softmax(t_2D_matrix predicted);
void		sigmoid(t_2D_matrix predicted);
void    	sigmoid_derv(t_2D_matrix a);
void		sm(int nb_row, int nb_col, float matrix[nb_row][nb_col]);

#endif