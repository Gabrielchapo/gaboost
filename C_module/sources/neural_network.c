#include "../include/binding.h"

void		dot_product_with_bias(t_2D_matrix a, t_2D_matrix b, t_2D_matrix bias, t_2D_matrix c)
{
	if (a.nb_col == b.nb_row && c.nb_row == a.nb_row && c.nb_col == b.nb_col)
	{
		for (int i = 0; i < c.nb_row * c.nb_col; i++)
		{
			c.values[i] = bias.values[i / c.nb_row];
			for (int j = 0; j < a.nb_col; j++)
				c.values[i] += a.values[(i / c.nb_col) * a.nb_col + j] * b.values[j * b.nb_col + (i % c.nb_col)];
		}
	}
	else
		PyErr_SetString(PyExc_ValueError, "Value error in dot product with bias");
}

PyObject	*neural_network_fit(PyObject *self, PyObject *args)
{
    /*****************************
	** PARSING X, Y, W and epoch *
	*****************************/

	PyObject* list_X;
	PyObject* list_Y;
    PyObject* list_W;
	PyObject* list_b;
    int nb_epoch;

  	if (!PyArg_ParseTuple(args, "O!O!O!O!i", &PyList_Type, &list_X, &PyList_Type, &list_Y, &PyList_Type, &list_W, &PyList_Type, &list_b, &nb_epoch)) 
    	return NULL;
	
	int nb_layers = PyList_Size(list_W);

    t_2D_matrix X = parsing(list_X, 0);
	t_2D_matrix Y = parsing(list_Y, 0);
	
	t_2D_matrix W[nb_layers];
	t_2D_matrix d_W[nb_layers];
	t_2D_matrix activation[nb_layers];
	t_2D_matrix tmp_activation[nb_layers];
	t_2D_matrix sigma[nb_layers];
	t_2D_matrix bias[nb_layers];
	t_2D_matrix d_bias[nb_layers];
	t_2D_matrix X_T = deep_copy_matrix(X);
	transpose_matrix(&X_T);

	for (int i = 0; i < nb_layers; i++)
	{
		W[i] = parsing(PyList_GetItem(list_W, i), 0);
		bias[i] = parsing(PyList_GetItem(list_b, i), 0);
		activation[i] = (i == 0) ? create_matrix_with_val(X.nb_row, W[i].nb_col, 0.0) : create_matrix_with_val(activation[i-1].nb_row, W[i].nb_col, 0.0);
		tmp_activation[i] = create_matrix_with_val(activation[i].nb_row, activation[i].nb_col, 0.0);
		d_W[i] = create_matrix_with_val(W[i].nb_row, W[i].nb_col, 0.0);
		d_bias[i] = create_matrix_with_val(bias[i].nb_row, bias[i].nb_col, 0.0);
	}
	for (int i = nb_layers - 1; i >= 0; i--)
		sigma[i] = (i == nb_layers - 1) ? create_matrix_with_val(X.nb_row, activation[i].nb_col, 0.0) : create_matrix_with_val(sigma[i+1].nb_row, activation[i].nb_col, 0.0);

	if (PyErr_Occurred()) return NULL;
	
	/************
	** TRAINING *
	************/

	//double loss;
	for (int epoch = 0; epoch < 3; epoch++)
	{
		// FEED FORWARD
		for (int i = 0; i < nb_layers; i++)
		{
			
			(i == 0) ? dot_product_with_bias(X, W[i], bias[i], activation[i]) : dot_product_with_bias(activation[i-1], W[i], bias[i], activation[i]);
			(i == nb_layers - 1) ? softmax(activation[i]) : sigmoid(activation[i]);
		}
		if (PyErr_Occurred()) return NULL;
		// BACK PROPAGATION 
		for (int i = nb_layers - 1; i >= 0; i--)
		{
			if (i == nb_layers - 1)
			{
				copy_matrix(sigma[i], activation[i]);
				diff_t_2D_matrix(sigma[i], Y);
				scalar_mult_t_2D_matrix(sigma[i], 1.0 / Y.nb_row);
			}
			else
			{
				copy_matrix(d_W[i+1], W[i+1]);
				transpose_matrix(&d_W[i+1]);
				dot_productt(sigma[i+1], d_W[i+1], sigma[i]);
				copy_matrix(tmp_activation[i], activation[i]);
				sigmoid_derv(tmp_activation[i]);
				mult_t_2D_matrix(sigma[i], tmp_activation[i]);
				if (PyErr_Occurred()) return NULL;
			}
		}
		if (PyErr_Occurred()) return NULL;
		for (int i = 0; i < nb_layers; i++)
		{
			if (i > 0)
			{	
				copy_matrix(tmp_activation[i-1], activation[i-1]);
				transpose_matrix(&tmp_activation[i-1]);
				transpose_matrix(&d_W[i]);
			}
			(i == 0) ? dot_productt(X_T, sigma[i], d_W[i]) : dot_productt(tmp_activation[i-1], sigma[i], d_W[i]);
			if (i > 0)
				transpose_matrix(&tmp_activation[i-1]);
			scalar_mult_t_2D_matrix(d_W[i], 0.3);
			diff_t_2D_matrix(W[i], d_W[i]);
			sum_in_t_2D_matrix(d_bias[i], sigma[i]);
			scalar_mult_t_2D_matrix(d_bias[i], 0.3);
			diff_t_2D_matrix(bias[i], d_bias[i]);
		}
		if (PyErr_Occurred()) return NULL;
		// LOSS
		/*
		for (int i = 0; i < nb_layers; i++)
		{
			(i == 0) ? dot_product_with_bias(X, W[i], bias[i], activation[i]) : dot_product_with_bias(activation[i-1], W[i], bias[i], activation[i]);
			(i == nb_layers - 1) ? softmax(activation[i]) : sigmoid(activation[i]);
		}
		for (int i = 0; i < activation[nb_layers - 1].nb_row * activation[nb_layers - 1].nb_col; i++)
		{
			activation[nb_layers - 1].values[i] = Y.values[i] * log(activation[nb_layers - 1].values[i]) + ((1 - Y.values[i]) * log(1.0 - activation[nb_layers - 1].values[i]));
			//printf("%f ", activation[nb_layers - 1].values[i]);
		}
		loss = 0;
		for (int i = 0; i < activation[nb_layers - 1].nb_row * activation[nb_layers - 1].nb_col; i++)
			loss += activation[nb_layers - 1].values[i];
		loss /= activation[nb_layers - 1].nb_row;
		loss *= -1;
		if (PyErr_Occurred()) return NULL;
		printf("epoch: %d, loss: %f\n", epoch, loss);*/
		printf("epoch: %d\n", epoch);
	}

	/**********
	** RETURN *
	**********/

	PyObject* python_val = PyList_New(nb_layers*2);
	for (int i = 0; i < nb_layers; i++)
		PyList_SetItem(python_val, i, create_PyObject_from_t_2D_matrix(W[i]));
	for (int i = nb_layers; i < nb_layers*2; i++)
		PyList_SetItem(python_val, i, create_PyObject_from_t_2D_matrix(bias[i - nb_layers]));
    return python_val;
}


PyObject	*neural_network_predict(PyObject *self, PyObject *args)
{
    /*****************************
	** PARSING X, Y, W and epoch *
	*****************************/

	PyObject* list_X;
    PyObject* list_W;
	PyObject* list_b;

  	if (!PyArg_ParseTuple(args, "O!O!O!", &PyList_Type, &list_X, &PyList_Type, &list_W, &PyList_Type, &list_b)) 
    	return NULL;
	
	int nb_layers = PyList_Size(list_W);

    t_2D_matrix X = parsing(list_X, 0);
	
	t_2D_matrix W[nb_layers];
	t_2D_matrix activation[nb_layers];
	t_2D_matrix bias[nb_layers];

	for (int i = 0; i < nb_layers; i++)
	{
		W[i] = parsing(PyList_GetItem(list_W, i), 0);
		bias[i] = parsing(PyList_GetItem(list_b, i), 0);
		activation[i] = (i == 0) ? create_matrix_with_val(X.nb_row, W[i].nb_col, 0.0) : create_matrix_with_val(activation[i-1].nb_row, W[i].nb_col, 0.0);
	}

	if (PyErr_Occurred()) return NULL;

	/************
	** TRAINING *
	************/

	for (int i = 0; i < nb_layers; i++)
	{
		(i == 0) ? dot_product_with_bias(X, W[i], bias[i], activation[i]) : dot_product_with_bias(activation[i-1], W[i], bias[i], activation[i]);
		(i == nb_layers - 1) ? softmax(activation[i]) : sigmoid(activation[i]);
	}

	/**********
	** RETURN *
	**********/

    return create_PyObject_from_t_2D_matrix(activation[nb_layers-1]);
}