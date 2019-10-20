#include "machine_learning.h"

void    clean_exit(t_env *env, char *error_message)
{
	if (env)
    {
        ft_bzero(env, sizeof(t_env));
        ft_memdel((void**)&env);
    }
    printf("%s", error_message);
    exit(0);
}
