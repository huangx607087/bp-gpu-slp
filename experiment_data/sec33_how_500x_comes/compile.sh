nvcc -O3 old_code_1.cu -DTIMUID=4
./a.out  #The first iteration costs 2 hours.

nvcc -O3 old_code_2.cu -DTIMUID=4
./a.out  #The first iteration costs 12 seconds.
