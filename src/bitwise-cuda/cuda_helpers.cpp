#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "cuda_helpers.h"

using namespace std;

void __checkCudaErrors(cudaError err, const char* file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

void __getLastCudaError(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

void printDeviceInfo(int cuda_device)
{
    // Print some CUDA device properties of the selected device.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    cout << "Name: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "MultiProcessor Count: " << prop.multiProcessorCount << endl;
    cout << "Clock Rate: " << prop.clockRate << " Hz" << endl;
    cout << "Warp Size: " << prop.warpSize << endl;
    cout << "Total Constant Memory: " << prop.totalConstMem << " bytes " << endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes " << endl;
    cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes " << endl;
    cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    cout << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
    cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "Regs Per Block: " << prop.regsPerBlock << endl;
    cout << "Memory Pitch: " << prop.memPitch << endl;
    cout << "Texture Alignment: " << prop.textureAlignment << endl;
    cout << "Device Overlap: " << prop.deviceOverlap << endl;
}

//
//// General GPU Device CUDA Initialization
//int gpuDeviceInit(int devID)
//{
//  int deviceCount;
//  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
//  if (deviceCount == 0) {
//    fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
//    exit(-1);
//  }
//  if (devID < 0)
//    devID = 0;
//  if (devID > deviceCount-1) {
//    fprintf(stderr, "\n");
//    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
//    fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
//    fprintf(stderr, "\n");
//    return -devID;
//  }
//
//  cudaDeviceProp deviceProp;
//  checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
//  if (deviceProp.major < 1) {
//    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
//    exit(-1);
//  }
//
//  checkCudaErrors( cudaSetDevice(devID) );
//  printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
//  return devID;
//}
//
//// This function returns the best GPU (with maximum GFLOPS)
//int gpuGetMaxGflopsDeviceId()
//{
//  int current_device   = 0, sm_per_multiproc = 0;
//  int max_compute_perf = 0, max_perf_device  = 0;
//  int device_count     = 0, best_SM_arch     = 0;
//  cudaDeviceProp deviceProp;
//
//  cudaGetDeviceCount( &device_count );
//  // Find the best major SM Architecture GPU device
//  while ( current_device < device_count ) {
//    cudaGetDeviceProperties( &deviceProp, current_device );
//    if (deviceProp.major > 0 && deviceProp.major < 9999) {
//      best_SM_arch = MAX(best_SM_arch, deviceProp.major);
//    }
//    current_device++;
//  }
//
//  // Find the best CUDA capable GPU device
//  current_device = 0;
//  while( current_device < device_count ) {
//    cudaGetDeviceProperties( &deviceProp, current_device );
//    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
//      sm_per_multiproc = 1;
//    } else {
//      sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
//    }
//
//    int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
//    if( compute_perf  > max_compute_perf ) {
//      // If we find GPU with SM major > 2, search only these
//      if ( best_SM_arch > 2 ) {
//        // If our device==dest_SM_arch, choose this, or else pass
//        if (deviceProp.major == best_SM_arch) {
//          max_compute_perf  = compute_perf;
//          max_perf_device   = current_device;
//        }
//      } else {
//        max_compute_perf  = compute_perf;
//        max_perf_device   = current_device;
//      }
//    }
//    ++current_device;
//  }
//  return max_perf_device;
//}
//
//// Initialization code to find the best CUDA Device
//int findCudaDevice(int argc, const char **argv)
//{
//  cudaDeviceProp deviceProp;
//  int devID = 0;
//  // If the command-line has a device number specified, use it
//  if (checkCmdLineFlag(argc, argv, "device")) {
//    devID = getCmdLineArgumentInt(argc, argv, "device=");
//    if (devID < 0) {
//      printf("Invalid command line parameters\n");
//      exit(-1);
//    } else {
//      devID = gpuDeviceInit(devID);
//      if (devID < 0) {
//        printf("exiting...\n");
//        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
//        exit(-1);
//      }
//    }
//  } else {
//    // Otherwise pick the device with highest Gflops/s
//    devID = gpuGetMaxGflopsDeviceId();
//    checkCudaErrors( cudaSetDevice( devID ) );
//    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
//    printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
//  }
//  return devID;
//}
