NAME = machine_learning

SRC_FILES = main.c \
			parsing.c \
			writing.c \
			clean_exit.c \
			tools.c \
			matrix.c \
			linear_gradientDescent.c \
			linear_normalEquation.c \
			linear_costFunction.c \
			linear_estimation.c \
			mean_normalization.c \
			logistic_regulated_gradientDescent.c \

OBJ_FILES = $(SRC_FILES:.c=.o)

SRC_PATH = srcs
OBJ_PATH = obj

SRC = $(addprefix $(SRC_PATH)/,$(SRC_FILES))
OBJ = $(addprefix $(OBJ_PATH)/,$(OBJ_FILES))

CC = gcc
CFLAGS = -Werror -Wall -Wextra

GREEN = \033[32m
RED = \033[31m
END = \033[0m

all: $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ 
	@echo "$(GREEN)$(NAME) ready to be executed ✓$(END)"

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c
	mkdir $(OBJ_PATH) 2> /dev/null || true
	$(CC) -I./include/ $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) 
	rmdir $(OBJ_PATH) 2> /dev/null || true
	@echo "$(RED).o files deleted ✗$(END)"

fclean: clean
	rm -f $(NAME) 
	@echo "$(RED)$(NAME) deleted ✗$(END)"

re: fclean all

.PHONY: all clean fclean re