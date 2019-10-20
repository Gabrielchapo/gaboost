#include "machine_learning.h"

void	parsing(t_env *env, char *src)
{
	int		fd;
	char	*line;

	if (!(fd = open(src, O_RDONLY)))
		clean_exit(NULL, "Error: Opening file has failed\n");
	while (get_next_line(fd, &line) > 0)
	{
		if (ft_strstr(line, "version"))
		{
			if (!(env->version = ft_strnew(5)))
				clean_exit(env, "Error: Memory allocation failed\n");
			env->version = ft_memcpy(env->version, line + 14, 5);
		}
		else if (ft_strstr(line, "option"))
		{
			if (!(env->option = ft_strnew(4)))
				clean_exit(env, "Error: Memory allocation failed\n");
			env->option = ft_memcpy(env->option, line + 13, 4);
		}
		else if (ft_strstr(line, "nb_training_examples"))
			env->nb_training_examples = atoi(line + 27);
		else if (ft_strstr(line, "nb_features"))
			env->nb_features = atoi(line + 18);
		else if (ft_strstr(line, "training_set"))
		{
			int j = 19;
			int k = 0;
			if (!(env->training_set = (double *)ft_memalloc(sizeof(double) * env->nb_training_examples * env->nb_features + 1)))
				clean_exit(env, "Error: Memory allocation failed\n");
			while (line[j] != '"' && line[j] != '\0')
			{
				env->training_set[k++] = atof(line + j);
				while (line[j] != ' ' && line[j] != '\0')
					j++;
				j++;
			}
		}
		else if (ft_strstr(line, "result"))
		{
			int j = 13;
			int k = 0;
			if (!(env->result = (double *)ft_memalloc(sizeof(double) * env->nb_training_examples + 1)))
				clean_exit(env, "Error: Memory allocation failed\n");
			while (line[j] != '"' && line[j] != '\0')
			{
				env->result[k++] = atof(line + j);
				while (line[j] != ' ' && line[j] != '\0')
					j++;
				j++;
			}
		}
		free(line);
	}
	if (close(fd) == -1)
		clean_exit(env, "Error: Closing file has failed\n");
}