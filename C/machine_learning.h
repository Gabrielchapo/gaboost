#ifndef MACHINE_LEARNING_H
# define MACHINE_LEARNING_H

# include <stdio.h>
# include <string.h>
# include <stdlib.h>
# include <math.h>
# include <unistd.h>
# include <fcntl.h>

# define BUFF_SIZE 4096
# define MAX_FD 100

typedef struct	s_matrix
{
	int			nb_column;
	int			nb_row;
	double		*value;
}				t_matrix;

typedef struct	s_env
{
	char		*version;
	char		*option;
	int			nb_training_examples;
	int			nb_features;
	double		*training_set;
	double		*result;
}				t_env;

void		parsing(t_env *env, char *src);
void    	clean_exit(t_env *env, char *error_message);
t_matrix	*linear_normalEquation(t_matrix *X, t_matrix *Y);
void		mean_normalization(t_matrix *matrix, int column);
double		linear_costFunction(t_matrix *matrix_a, t_matrix *matrix_b, t_matrix *thetas);
double		linear_estimation(t_matrix *X, int line, t_matrix *thetas);
t_matrix	*linear_gradientDescent(t_matrix *matrix_a, t_matrix *matrix_b, double gradient, int nb_iterations);
t_matrix	*logistic_regulated_gradientDescent(t_matrix *X, t_matrix *Y, t_matrix *lambda, double gradient, int nb_iterations);
void		writing(char *file, t_env *env, t_matrix *thetas);


/*
** tools.c
*/

void	*ft_memalloc(size_t size);
size_t	ft_strlen(const char *s);
void	ft_bzero(void *b, size_t len);
void	*ft_memset(void *b, int c, size_t len);
void	*ft_memcpy(void *dst, const void *src, size_t n);
int		ft_strcmp(const char *s1, const char *s2);
void	ft_memdel(void **ap);
char	*ft_strnew(size_t size);
char	*ft_strjoin(char const *s1, char const *s2);
int		get_next_line(const int fd, char **line);
char	*ft_strdup(const char *s1);
char	*ft_strstr(const char *haystack, const char *needle);


/*
** matrix.c
*/
t_matrix	*add_matrixes(t_matrix *matrix_a, t_matrix *matrix_b);
t_matrix	*sub_matrixes(t_matrix *matrix_a, t_matrix *matrix_b);
t_matrix	*multiply_matrixes(t_matrix *matrix_a, t_matrix *matrix_b);
t_matrix	*scalar_mult(t_matrix *matrix_a, double scalar);
t_matrix	*scalar_div(t_matrix *matrix_a, double scalar);
t_matrix	*create_matrix(double *value, int nb_column, int nb_row);
t_matrix	*transpose_matrix(t_matrix *matrix_a);
t_matrix	*inverse_matrix(t_matrix *matrix_a);
void		print_matrix(t_matrix *matrix);
void		free_matrix(t_matrix *matrix);
double		get_value(t_matrix *matrix, int column, int row);

#endif
