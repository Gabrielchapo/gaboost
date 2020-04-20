#include "../include/binding.h"

PyObject	*create_PyObject_from_t_2D_matrix(t_2D_matrix a)
{
	PyObject* python_val = PyList_New(a.nb_row);
	
	for (int i = 0; i < a.nb_row; i++)
	{
		if (a.nb_col == 1)
			PyList_SetItem(python_val, i, Py_BuildValue("f", a.values[i]));
		else
		{
			PyObject* sub = PyList_New(a.nb_col);
			for (int j = 0; j < a.nb_col; j++)
				PyList_SetItem(sub, j, Py_BuildValue("f", a.values[i * a.nb_col + j]));
			PyList_SetItem(python_val, i, sub);
		}
	}
	return python_val;
}

t_2D_matrix	parsing(PyObject *list, int bias)
{
	t_2D_matrix matrix;

	// Set number of observations
	matrix.nb_row = PyList_Size(list);

	// Get the number of parameter for X, and add one for the bias
	matrix.nb_col = PyList_Size(PyList_GetItem(list, 0));
	matrix.nb_col += bias;

	// MALLOC
	if (!(matrix.values = (double*)malloc(sizeof(double) * matrix.nb_col * matrix.nb_row)))
	{
		PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
		return matrix;
	}

	// PARSING
	for (int i = 0; i < matrix.nb_row; i++)
	{
		PyObject *sublist = PyList_GetItem(list, i);
		if (PyList_Size(sublist) + bias != matrix.nb_col)
		{
			PyErr_SetString(PyExc_ValueError, "Number of parameters must be the same for each obervations");
			return matrix;
		}
		for (int j = 0; j < matrix.nb_col - bias; j++)
			matrix.values[i * matrix.nb_col + j] = PyFloat_AsDouble(PyList_GetItem(sublist, j));
		if (bias == 1)
			matrix.values[i * matrix.nb_col + matrix.nb_col - bias] = 1;
  	}
	return matrix;
}

void	parsingg(PyObject *list, int bias, int nb_row, int nb_col, float matrix[nb_row][nb_col])
{
	for (int i = 0; i < nb_row; i++)
	{
		PyObject *sublist = PyList_GetItem(list, i);
		for (int j = 0; j < nb_col - bias; j++)
			matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(sublist, j));
		if (bias == 1)
			matrix[i][nb_col - 1] = 1;
	}
}

PyObject	*create_return(int nb_row, int nb_col, float matrix[nb_row][nb_col])
{
	PyObject* python_val = PyList_New(nb_row);
	for (int i = 0; i < nb_row; i++)
	{
		if (nb_col == 1)
			PyList_SetItem(python_val, i, Py_BuildValue("f", matrix[i][0]));
		else
		{
			PyObject* sub = PyList_New(nb_col);
			for (int j = 0; j < nb_col; j++)
				PyList_SetItem(sub, j, Py_BuildValue("f", matrix[i][j]));
			PyList_SetItem(python_val, i, sub);
		}
	}
	return python_val;
}