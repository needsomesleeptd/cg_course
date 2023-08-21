#!/bin/bash


cat << EOF > /tmp/cudaComputeVersion.cu
#include <stdio.h>
int main()
{
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop,0);
int v = prop.major * 10 + prop.minor;
printf("-gencode arch=compute_%d,code=sm_%d\n",v,v);
}
EOF


/usr/local/cuda/bin/nvcc /tmp/cudaComputeVersion.cu -o /tmp/cudaComputeVersion
/tmp/cudaComputeVersion
rm /tmp/cudaComputeVersion.cu
rm /tmp/cudaComputeVersion