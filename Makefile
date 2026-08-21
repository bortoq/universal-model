# top-level Makefile for src/vm1
VM1_DIR=src/vm1

all: vm1 sort2

vm1:
	cc -Wall -Wextra -std=c11 -O2 $(VM1_DIR)/vm1.c -o $(VM1_DIR)/vm1

sort2:
	cc -Wall -Wextra -std=c11 -O2 $(VM1_DIR)/sort2.c -o $(VM1_DIR)/sort2

sorter: sort2
	./$(VM1_DIR)/sort2 65520 > $(VM1_DIR)/sorter.txt

clean:
	rm -f $(VM1_DIR)/vm1 $(VM1_DIR)/sort2 $(VM1_DIR)/sorter.txt

run: vm1 sorter
	./$(VM1_DIR)/vm1 $(VM1_DIR)/sorter.txt < /dev/stdin > /dev/stdout
