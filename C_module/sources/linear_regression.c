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
	t_2D_matrix X_T = transposed_matrix(X);
	t_2D_matrix Y = parsing(list_Y, 0);
	t_2D_matrix W = initialize_weights(X.nb_col, Y.nb_col);

	if (PyErr_Occurred()) return NULL;

	/************
	** TRAINING *
	************/

	for (int epoch = 0; epoch < 3000; epoch++)
	{
		// GET Gradient vector
		t_2D_matrix predicted = dot_product(X, W);
		if (PyErr_Occurred()) return NULL;
		for (int i = 0; i < predicted.nb_row * predicted.nb_col; i++)
			predicted.values[i] -= Y.values[i];
		t_2D_matrix d_W = dot_product(X_T, predicted);
		// UPDATE weights W
		for (int i = 0; i < W.nb_row * W.nb_col; i++)
			W.values[i] -= ((0.05 / X.nb_row) * d_W.values[i]);

		//printf("epoch %d, cost:%f\n", epoch, mean_square_error(X,Y,W));
	}

  	return create_PyObject_from_t_2D_matrix(W);
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

	if (PyErr_Occurred()) return NULL;

	/***********
	** PREDICT *
	***********/

	t_2D_matrix predicted = dot_product(X, W);
	
	return create_PyObject_from_t_2D_matrix(predicted);
}