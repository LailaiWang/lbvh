all:
	nvcc -std=c++17 -O2  main.cu -o Run -I /usr/local/cuda-11.7/include/ -I ./lbvh -L /usr/local/cuda-11.7/lib64/ -lcudart -rdc=true  --extended-lambda 
