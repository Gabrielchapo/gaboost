#include "../include/binding.h"

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
	t_2D_matrix activation[nb_layers];
	t_2D_matrix sigma[nb_layers];
	t_2D_matrix bias[nb_layers];

	for (int i = 0; i < nb_layers; i++)
	{
		W[i] = parsing(PyList_GetItem(list_W, i), 0);
		bias[i] = parsing(PyList_GetItem(list_b, i), 0);
	}
	
	for (int i = 0; i < nb_layers; i++)
		printf("layer %d,  shape : (%d, %d)\n",i, bias[i].nb_row, bias[i].nb_col);

	if (PyErr_Occurred()) return NULL;

	/************
	** TRAINING *
	************/
	nb_epoch = 1;
	double loss;
	for (int epoch = 0; epoch < nb_epoch; epoch++)
	{
		// FEEDFORWARD
		t_2D_matrix tmp = X;
		for (int i = 0; i < nb_layers; i++)
		{
			activation[i] = dot_product_with_bias(tmp, W[i], bias[i]);
			//print_t_2D_matrix(activation[i]);
			if (i < nb_layers - 1)
				sigmoid(activation[i]);
			else
				softmax(activation[i]);
			tmp = activation[i];
		}
		sigma[nb_layers - 1] = scalar_mult_t_2D_matrix(diff_t_2D_matrix(activation[nb_layers-1], Y), 1.0/Y.nb_row);
		//print_t_2D_matrix(sigma[nb_layers - 1]);
		for (int i = nb_layers - 2; i >= 0; i--)
		{
			tmp = dot_product(sigma[i+1], transposed_matrix(W[i+1]));
			t_2D_matrix tmp2 = sigmoid_derv(activation[i]);
			sigma[i] = mult_t_2D_matrix(tmp, tmp2);
			//print_t_2D_matrix(sigma[i]);
			if (PyErr_Occurred()) return NULL;
		}
		tmp = transposed_matrix(X);
		for (int i = 0; i < nb_layers; i++)
		{
			W[i] = diff_t_2D_matrix(W[i], scalar_mult_t_2D_matrix(dot_product(tmp, sigma[i]), 0.3));
			bias[i] = diff_t_2D_matrix(bias[i], scalar_mult_t_2D_matrix(sum_in_t_2D_matrix(sigma[i]), 0.3));
			tmp = transposed_matrix(activation[i]);
		}
		if (PyErr_Occurred()) return NULL;
		tmp = X;
		for (int i = 0; i < nb_layers; i++)
		{
			activation[i] = dot_product_with_bias(tmp, W[i], bias[i]);
			
			if (i < nb_layers - 1)
				sigmoid(activation[i]);
			else
				softmax(activation[i]);
			tmp = activation[i];
		}
		
		if (PyErr_Occurred()) return NULL;
		for (int i = 0; i < tmp.nb_row * tmp.nb_col; i++)
			tmp.values[i] = Y.values[i] * log(tmp.values[i]) + ((1 - Y.values[i]) * log(1.0 - tmp.values[i]));
		
		loss = 0;
		for (int i = 0; i < tmp.nb_row * tmp.nb_col; i++)
			loss += tmp.values[i];
		loss /= tmp.nb_row;
		loss *= -1;
		printf("epoch: %d, loss: %f\n", epoch, loss);
	}

	/**********
	** RETURN *
	**********/

	PyObject* python_val = PyList_New(nb_layers*2);
	for (int i = 0; i < nb_layers; i++)
		PyList_SetItem(python_val, i, create_PyObject_from_t_2D_matrix(W[i]));
	for (int i = nb_layers; i < nb_layers*2; i++)
		PyList_SetItem(python_val, i, create_PyObject_from_t_2D_matrix(bias[i]));
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
	}
	
	for (int i = 0; i < nb_layers; i++)
		printf("layer %d,  shape : (%d, %d)\n",i, bias[i].nb_row, bias[i].nb_col);

	if (PyErr_Occurred()) return NULL;

	/************
	** TRAINING *
	************/
	t_2D_matrix tmp = X;
	for (int i = 0; i < nb_layers; i++)
	{
		activation[i] = dot_product_with_bias(tmp, W[i], bias[i]);
		if (i < nb_layers - 1)
			sigmoid(activation[i]);
		else
			softmax(activation[i]);
		tmp = activation[i];
	}

	/**********
	** RETURN *
	**********/

    return create_PyObject_from_t_2D_matrix(tmp);
}