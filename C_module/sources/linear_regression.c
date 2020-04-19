#include "../include/binding.h"

PyObject	*linear_regression_fit(PyObject *self, PyObject *args)
{
	/*******************
	** PARSING X AND Y *
	*******************/

	PyObject* list_X;
	PyObject* list_Y;

  	if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &list_X, &PyList_Type, &list_Y)) 
    	return NULL;

	t_2D_matrix X = parsing(list_X, 1);
	t_2D_matrix Y = parsing(list_Y, 0);

	/********************
	** USEFUL VARIABLES *
	********************/

	t_2D_matrix X_T = deep_copy_matrix(X);
	transpose_matrix(&X_T);
	t_2D_matrix W = create_matrix_with_val(X.nb_col, Y.nb_col, 1.0);
	t_2D_matrix predicted = create_matrix_with_val(X.nb_row, W.nb_col, 0.0);
	t_2D_matrix d_W = create_matrix_with_val(X_T.nb_row, predicted.nb_col, 0.0);

	if (PyErr_Occurred()) return NULL;

	/************
	** TRAINING *
	************/
	double initial_cost = -1;
	double cost = -2;

	for (int epoch = 0; fabs((cost-initial_cost)/initial_cost) > 0.001 ; epoch++)
	{
		initial_cost = cost;
		dot_productt(X, W, predicted);

		// prediction - target values
		for (int i = 0; i < predicted.nb_row * predicted.nb_col; i++)
			predicted.values[i] -= Y.values[i];
		
		// GET Gradient vector d_W
		dot_productt(X_T, predicted, d_W);

		// UPDATE weights W
		for (int i = 0; i < W.nb_row * W.nb_col; i++)
			W.values[i] -= ((0.05 / X.nb_row) * d_W.values[i]);
		
		if (PyErr_Occurred()) return NULL;
		cost = mean_square_error(X, Y, W);
		printf("epoch %d, cost:%f\n", epoch, cost);
	}

	/*********************************************
	** CREATING PYTHON OBJECT AND FREE VARIABLES *
	*********************************************/
	
	PyObject *list_W = create_PyObject_from_t_2D_matrix(W);
	free_matrix(W);
	free_matrix(X);
	free_matrix(Y);
	free_matrix(d_W);
	free_matrix(predicted);
	free_matrix(X_T);

  	return list_W;
}

PyObject	*linear_regression_predict(PyObject *self, PyObject *args)
{
	/*******************
	** PARSING X AND W *
	*******************/

	PyObject* list_X;
	PyObject* list_W;

  	if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &list_X, &PyList_Type, &list_W)) 
    	return NULL;
	
	t_2D_matrix X = parsing(list_X, 1);
	t_2D_matrix W = parsing(list_W, 0);

	t_2D_matrix predicted = create_matrix_with_val(X.nb_row, W.nb_col, 0.0);

	if (PyErr_Occurred()) return NULL;

	/**************
	** PREDICTING *
	**************/

	dot_productt(X, W, predicted);
	
	/*********************************************
	** CREATING PYTHON OBJECT AND FREE VARIABLES *
	*********************************************/
	
	PyObject *list_predicted = create_PyObject_from_t_2D_matrix(predicted);
	free_matrix(W);
	free_matrix(X);
	free_matrix(predicted);
	
  	return list_predicted;
}