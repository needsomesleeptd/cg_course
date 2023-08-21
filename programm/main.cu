#include "mainwindow.h"
#include <iostream>
#include <QApplication>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
		          file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

int main(int argc, char *argv[])
{
	int num_pixels = 12*12;
	size_t fb_size = num_pixels*sizeof(VecD3);
	VecD3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
	int cudaVersion;
	cudaRuntimeGetVersion(&cudaVersion);
	std::cout << "CUDA Version: " << cudaVersion / 1000 << "."
	          << (cudaVersion % 1000) / 10 << std::endl;
	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}
