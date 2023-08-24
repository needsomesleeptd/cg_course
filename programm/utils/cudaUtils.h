//
// Created by Андрей on 23.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_UTILS_CUDAUTILS_H_
#define LAB_03_CG_COURSE_PROGRAMM_UTILS_CUDAUTILS_H_

#include <cstdio>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cassert>
#include <iostream>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__device__  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)  assert(0);
	}
}

#define cpuErrorCheck(val) cpuAssert( (val), #val, __FILE__, __LINE__ )

inline void cpuAssert(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
		          file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}
#endif //LAB_03_CG_COURSE_PROGRAMM_UTILS_CUDAUTILS_H_
