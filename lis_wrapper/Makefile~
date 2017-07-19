o_files = main.o solver_wrapper.o
cc_com = gcc -c -O3 -std=gnu11 -fopenmp -fomit-frame-pointer -DHAVE_CONFIG_H -I. -I/home/zhangji/stokes_flow/build/lis/include -llis -lm

lis_wrapper : $(o_files)
	gcc -fopenmp -o lis_wrapper -lm $(o_files) -L/home/zhangji/stokes_flow/build/lis/lib -llis -lm
	
main.o : main.c
	$(cc_com) main.c

solver_wrapper.o : solver_wrapper.c
	$(cc_com) solver_wrapper.c

.PHONY : clean
clean: 
	-rm lis_wrapper $(o_files)

