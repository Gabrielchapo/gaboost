# include "machine_learning.h"

void	writing(char *file, t_env *env, t_matrix *thetas)
{
	int		fd;

	(void)thetas;
	if ((fd = open(file, O_CREAT | O_WRONLY | O_TRUNC, 0600)) == -1)
		clean_exit(env, "Error: Opening/Creating new file has failed\n");
	dprintf(fd, "{\n\t\"version\" : \"%s\",\n\t\"nb_thetas\" : \"%d\",\n\t\"thetas\" : \"", env->version, thetas->nb_row);
	for (int i = 0 ; i < thetas->nb_row ; i++)
	{
		dprintf(fd, "%f", thetas->value[i]);
		if (i + 1 != thetas->nb_row)
			dprintf(fd, " ");
	}
	dprintf(fd, "\"\n}\n");
	if (close(fd) == -1)
		clean_exit(env, "Error: Closing new file has failed\n");
}
