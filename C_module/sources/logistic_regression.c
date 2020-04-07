#include "../include/binding.h"

void		softmax(t_2D_matrix predicted)
{
	double the_sum;

	for (int i = 0; i < predicted.nb_row; i++)
	{
		the_sum = 0;
		for (int j = 0 ; j < predicted.nb_col ; j++)
			the_sum += exp(predicted.values[i * predicted.nb_col + j]);
		for (int j = 0 ; j < predicted.nb_col ; j++)
			predicted.values[i * predicted.nb_col + j] = exp(predicted.values[i * predicted.nb_col + j]) / the_sum;
	}
}

double		loss(t_2D_matrix X, t_2D_matrix Y, t_2D_matrix W)
{
	double loss = 0;
	t_2D_matrix predicted = dot_product(X, W);
	softmax(predicted);
	for (int i = 0; i < predicted.nb_row * predicted.nb_col; i++)
		predicted.values[i] = Y.values[i] * log(predicted.values[i]) + ((1 - Y.values[i]) * log(1.0 - predicted.values[i]));
	double sum[predicted.nb_row];
	for (int i = 0; i < predicted.nb_row; i++)
	{
		sum[i] = 0;
		for (int j = 0 ; j < predicted.nb_col ; j++)
			sum[i] += predicted.values[i * predicted.nb_col + j];
		loss += sum[i];
	}
	loss /= predicted.nb_row;
	loss *= -1;
	free(predicted.values);
	return loss;
}

PyObject	*logistic_regression_fit(PyObject *self, PyObject *args)
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

	for (int epoch=0; epoch < 3000; epoch++)
	{
		// GET Gradient vector
		t_2D_matrix predicted = dot_product(X, W);
		softmax(predicted);
		if (PyErr_Occurred()) return NULL;
		for (int i = 0; i < predicted.nb_row * predicted.nb_col; i++)
			predicted.values[i] -= Y.values[i];
		t_2D_matrix d_W = dot_product(X_T, predicted);

		// UPDATE weights W
		for (int i = 0; i < W.nb_row * W.nb_col; i++)
			W.values[i] -= ((0.05 / X.nb_row) * d_W.values[i]);
		printf("epoch %d, loss %f\n", epoch, loss(X, Y, W));	
	}

  	return create_PyObject_from_t_2D_matrix(W);
}

PyObject	*logistic_regression_predict(PyObject *self, PyObject *args)
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
	softmax(predicted);

	if (PyErr_Occurred()) return NULL;
	
	return create_PyObject_from_t_2D_matrix(predicted);
}
