main: main.cc 
	g++-6 main.cc -o main -fopenmp 
	./main > imagem.ppm
