#include "machine_learning.h"

int		check_extensions(char *src, char *dest)
{
	int i = ft_strlen(src) - 5;
	int	j = ft_strlen(dest) - 5;

	if (i <= 0 || j <= 0 || ft_strcmp(src + i, ".JSON") || ft_strcmp(dest + j, ".JSON"))
		return (0);
	else
		return (1);
}

int		main(int argc, char **argv)
{
	t_env *env;
	t_matrix *X;
	t_matrix *Y;
	t_matrix *Thetas;

	// Check right Usage
	if (argc != 3)
		clean_exit(NULL, "Usage : ./machine_learning <src_file.JSON> <dest_file.JSON>\n");
	if (!check_extensions(argv[1], argv[2]))
		clean_exit(NULL, "Error : wrong extension (.JSON) : Usage : ./machine_learning <src_file.JSON> <dest_file.JSON>\n");
	
	// fill env structure
	if (!(env = ft_memalloc(sizeof(t_env))))
		clean_exit(env, "Error: Memory allocation failed\n");
	parsing(env, argv[1]);

	if (!(X = create_matrix(env->training_set, env->nb_features, env->nb_training_examples)))
		clean_exit(env, "Error: Memory allocation failed\n");
	//print_matrix(X);
	//mean_normalization(X, 0);
	//mean_normalization(X, 1);

	if (!(Y = create_matrix(env->result, 1, env->nb_training_examples)))
		clean_exit(env, "Error: Memory allocation failed\n");
	//print_matrix(Y);
	//mean_normalization(Y, 0);
	/*
	if (!(Thetas = linear_normalEquation(X, Y)))
		clean_exit(env, "Error: Memory allocation failed\n");
	print_matrix(Thetas);

	printf("cost function for normal equation :%f\n", linear_costFunction(X,Y, Thetas));
	printf("Estimation for normal equation :%f\n", linear_estimation(X, 1, Thetas));

	if (!(Thetas = linear_gradientDescent(X, Y, 0.05, 350)))
		clean_exit(env, "Error: Memory allocation failed\n");
	print_matrix(Thetas);

	printf("cost function for gradient descent :%f\n", linear_costFunction(X,Y, Thetas));
	printf("Estimation for gradient descent :%f\n", linear_estimation(X, 1, Thetas));
	*/
	t_matrix *lambda;
	double value[2];

	value[0] = 0;
	value[1] = 0;
	lambda = create_matrix(value, 1, 2);

	if (!(Thetas = logistic_regulated_gradientDescent(X, Y, lambda, 0.03, 10)))
		clean_exit(env, "Error: Memory allocation failed\n");
	print_matrix(Thetas);
	writing(argv[2], env, Thetas);

	clean_exit(env, "SUCCESS\n");
}
