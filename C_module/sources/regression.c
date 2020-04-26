#include "../include/binding.h"

/*
** 0 - linear regression
** 1 - logistic regression
*/

PyObject	*regression_fit(PyObject *self, PyObject *args)
{
  	PyObject* list_X;
	PyObject* list_Y;
    uint8_t opt;

  	if (!PyArg_ParseTuple(args, "O!O!b", &PyList_Type, &list_X, &PyList_Type, &list_Y, &opt)) 
    	return NULL;

	t_size size_X;
	size_X.nb_row = PyList_Size(list_X);
	size_X.nb_col = PyList_Size(PyList_GetItem(list_X, 0)) + 1;
	float X[size_X.nb_row][size_X.nb_col];
	parsingg(list_X, 1, size_X.nb_row, size_X.nb_col, X);

	t_size size_Y;
	size_Y.nb_row = PyList_Size(list_Y);
	size_Y.nb_col = PyList_Size(PyList_GetItem(list_Y, 0));
	float Y[size_Y.nb_row][size_Y.nb_col];
	parsingg(list_Y, 0, size_Y.nb_row, size_Y.nb_col, Y);
	
	t_size size_W;
	size_W.nb_row = size_X.nb_col;
	size_W.nb_col = size_Y.nb_col;
	float W[size_W.nb_row][size_W.nb_col];

	float d_W[size_X.nb_col][size_Y.nb_col];

	t_size size_pred;
	size_pred.nb_row = size_X.nb_row;
	size_pred.nb_col = size_Y.nb_col;
	float pred[size_pred.nb_row][size_pred.nb_col];

	for (int i = 0; i < size_W.nb_row; i++)
	{
		for (int j = 0; j < size_W.nb_col; j++)
			W[i][j] = 1;
	}

	double initial_cost = -1;
	double cost = -2;

	for (int epoch = 0; fabs((cost-initial_cost)/initial_cost) > 0.00000000001 ; epoch++)
	{
		initial_cost = cost;
		dot(size_X.nb_row, size_X.nb_col, X, size_W.nb_row, size_W.nb_col, W, pred);

        (opt == 1) ? sm(size_pred.nb_row, size_pred.nb_col, pred) : 0;
        
		// prediction - target values
		for (int i = 0 ; i < size_pred.nb_row; i++)
		{
			for (int j = 0; j < size_pred.nb_col; j++)
				pred[i][j] -= Y[i][j];
		}
		
		// GET Gradient vector d_W
		dot_T(size_X.nb_row, size_X.nb_col, X, size_pred.nb_row, size_pred.nb_col, pred, d_W);

		// UPDATE weights W
		for (int i=0;i<size_W.nb_row;i++)
		{
			for (int j=0;j<size_W.nb_col;j++)
				W[i][j] -= ((0.05 / size_X.nb_row) * d_W[i][j]);
		}
		
		if (PyErr_Occurred()) return NULL;
        cost = (opt == 1) ? ce(size_X.nb_row, size_X.nb_col, X, size_Y.nb_row, size_Y.nb_col, Y, W, pred)
            : mse(size_X.nb_row, size_X.nb_col, X, size_Y.nb_row, size_Y.nb_col, Y, W);
		//printf("epoch %d, cost:%f\n", epoch, cost);
	}

  	return create_return(size_W.nb_row, size_W.nb_col, W);
}

PyObject	*regression_predict(PyObject *self, PyObject *args)
{
	PyObject* list_X;
	PyObject* list_W;
    uint8_t opt;

  	if (!PyArg_ParseTuple(args, "O!O!b", &PyList_Type, &list_X, &PyList_Type, &list_W, &opt)) 
    	return NULL;

	t_size size_X;
	size_X.nb_row = PyList_Size(list_X);
	size_X.nb_col = PyList_Size(PyList_GetItem(list_X, 0)) + 1;
	float X[size_X.nb_row][size_X.nb_col];
	parsingg(list_X, 1, size_X.nb_row, size_X.nb_col, X);

	t_size size_W;
	size_W.nb_row = PyList_Size(list_W);
	size_W.nb_col = PyList_Size(PyList_GetItem(list_W, 0));
	float W[size_W.nb_row][size_W.nb_col];
	parsingg(list_W, 0, size_W.nb_row, size_W.nb_col, W);
	
	t_size size_pred;
	size_pred.nb_row = size_X.nb_row;
	size_pred.nb_col = size_W.nb_col;
	float pred[size_pred.nb_row][size_pred.nb_col];

	dot(size_X.nb_row, size_X.nb_col, X, size_W.nb_row, size_W.nb_col, W, pred);
    (opt == 1) ? sm(size_pred.nb_row, size_pred.nb_col, pred) : 0;

	return create_return(size_pred.nb_row, size_pred.nb_col, pred);
}