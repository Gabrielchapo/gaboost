#include "machine_learning.h"

void	ft_memdel(void **ap)
{
	if (!ap)
		return ;
	free(*ap);
	*ap = NULL;
}

void		*ft_memalloc(size_t size)
{
	void	*outptr;

	if (!(outptr = (void *)malloc(size)))
		return (NULL);
	ft_memset(outptr, 0, size);
	return (outptr);
}

void	*ft_memset(void *b, int c, size_t len)
{
	unsigned char	l;
	size_t			i;
	unsigned char	*bchar;

	bchar = (unsigned char *)b;
	l = (unsigned char)c;
	i = 0;
	while (i < len)
	{
		*bchar++ = l;
		i++;
	}
	return (b);
}

char		*ft_strstr(const char *haystack, const char *needle)
{
	int i;
	int j;

	i = 0;
	j = 0;
	if (needle[0] == '\0')
		return ((char *)haystack);
	while (haystack[i] != '\0')
	{
		j = 0;
		while (needle[j] != '\0' && haystack[i + j] != '\0'
				&& needle[j] == haystack[i + j])
			j++;
		if (needle[j] == '\0')
			return ((char *)(haystack + i));
		i++;
	}
	return (0);
}

void				*ft_memcpy(void *dst, const void *src, size_t n)
{
	size_t				i;
	unsigned char		*dchar;
	const unsigned char *schar;

	if (!dst && !src)
		return (dst);
	i = 0;
	dchar = (unsigned char *)dst;
	schar = (const unsigned char*)src;
	while (i < n)
	{
		dchar[i] = schar[i];
		i++;
	}
	return (dst);
}
char			*ft_strcpy(char *dst, const char *src)
{
	int	i;

	i = 0;
	while (src[i] != '\0')
	{
		dst[i] = src[i];
		i++;
	}
	dst[i] = '\0';
	return (dst);
}

void				ft_bzero(void *b, size_t len)
{
	ft_memset(b, '\0', len);
}

char	*ft_strnew(size_t size)
{
	char	*out;

	if (!(out = (char *)malloc(sizeof(char) * (size + 1))))
		return (NULL);
	ft_bzero(out, size + 1);
	return (out);
}

size_t		ft_strlen(const char *s)
{
	size_t i;

	i = 0;
	while (s[i] != 0)
		i++;
	return (i);
}

int		ft_strcmp(const char *s1, const char *s2)
{
	while (*s1 && *s2 && (unsigned char)*s1 == (unsigned char)*s2)
	{
		s1++;
		s2++;
	}
	return ((unsigned char)*s1 - (unsigned char)*s2);
}

char	*ft_strchr(const char *s, int c)
{
	unsigned char	cchar;

	cchar = (unsigned char)c;
	while (*s)
	{
		if (*s == cchar)
			return ((char *)s);
		s++;
	}
	if (c == 0)
		return ((char *)s);
	else
		return (NULL);
}

int				free_static(int fd, char **str)
{
	int i;

	i = 0;
	if (fd == -2)
	{
		while (i < MAX_FD)
		{
			ft_memdel((void **)&str[i]);
			i++;
		}
		return (1);
	}
	return (0);
}


char	*ft_strdup(const char *s1)
{
	char	*out;
	int		l;
	int		i;

	i = 0;
	out = 0;
	l = ft_strlen(s1);
	if (!(out = (char *)malloc(sizeof(char) * (l + 1))))
		return (NULL);
	while (s1[i])
	{
		out[i] = s1[i];
		i++;
	}
	out[i] = 0;
	return (out);
}

char	*ft_strjoin(char const *s1, char const *s2)
{
	size_t	i;
	size_t	j;
	char	*fresh;

	i = 0;
	j = 0;
	if (!s1 || !s2 || !(fresh = ft_strnew(ft_strlen(s1) + ft_strlen(s2))))
		return (NULL);
	while (s1[i])
	{
		fresh[i] = s1[i];
		i++;
	}
	while (s2[j])
	{
		fresh[i + j] = s2[j];
		j++;
	}
	fresh[i + j] = 0;
	return (fresh);
}

static int		ft_free_return(char **gnl, int val)
{
	if (*gnl)
		ft_memdel((void **)gnl);
	return (val);
}

static int		ft_complete_line(char **gnl, int fd, char *buf, int *r)
{
	char		*temp;

	while (!ft_strchr(gnl[fd], '\n') && (*r = read(fd, buf, BUFF_SIZE)) > 0)
	{
		if ((int)ft_strlen(buf) != *r)
			return (ft_free_return(&gnl[fd], 0));
		temp = gnl[fd];
		gnl[fd] = ft_strjoin(gnl[fd], buf);
		free(temp);
		if (!gnl[fd])
			return (0);
		ft_bzero(buf, BUFF_SIZE + 1);
	}
	return (1);
}

int				get_next_line(const int fd, char **line)
{
	static char	*gnl[MAX_FD] = {0};
	char		buf[BUFF_SIZE + 1];
	int			r;
	char		*temp;

	r = 0;
	if (free_static(fd, gnl) || fd < 0 || fd >= MAX_FD
	|| !line || BUFF_SIZE <= 0)
		return (-1);
	if (!(gnl[fd] = !gnl[fd] ? ft_strnew(0) : gnl[fd]))
		return (-1);
	*line = 0;
	ft_bzero(buf, BUFF_SIZE + 1);
	if (!ft_complete_line(gnl, fd, buf, &r))
		return (-1);
	if (r == -1 || (!r && !*gnl[fd]))
		return (ft_free_return(&gnl[fd], r == -1 ? -1 : 0));
	temp = ft_strchr(gnl[fd], '\n');
	r = !temp ? (ft_strlen(gnl[fd])) : temp - gnl[fd];
	if ((*line = ft_strnew(r)))
		ft_memcpy(*line, gnl[fd], r);
	ft_strcpy(gnl[fd], temp ? temp + 1 : "");
	if (!*line)
		return (ft_free_return(&gnl[fd], -1));
	return (1);
}
