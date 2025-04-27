/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This code and more samples can be found at https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <opencv2/opencv.hpp> // Include OpenCV for image processing
#include <random>

#define HANDLE_ERROR(x)                                               \
  {                                                                   \
    const auto err = x;                                               \
    if (err != CUTENSOR_STATUS_SUCCESS)                               \
    {                                                                 \
      printf("Error: %s\n", cutensorGetErrorString(err));              \
      return err;                                                     \
    }                                                                 \
  };

#define HANDLE_CUDA_ERROR(x)                                      \
  {                                                               \
    const auto err = x;                                           \
    if (err != cudaSuccess)                                      \
    {                                                             \
      printf("Error: %s\n", cudaGetErrorString(err));              \
      return err;                                                 \
    }                                                             \
  };

using namespace std;
using namespace cv;

void printFloatArray(float *x, int numElements) {
  for (int i = 0; i < numElements; i++) std::cout << x[i] << " ";
  std::cout << std::endl;
}

struct GPUTimer {
  GPUTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_, 0); }

  float seconds() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

 private:
  cudaEvent_t start_, stop_;
};

// Function to generate a random image
cv::Mat generateRandomImage(int width, int height) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255); // Range for pixel values

    // Create a Mat object to store the image (grayscale)
    cv::Mat image(height, width, CV_8UC1);

    // Fill the image with random pixel values
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at<uchar>(y, x) = distrib(gen);
        }
    }

    return image;
}


int main(int argc, char** argv) {

// Image dimensions
    int width = 512;
    int height = 512;

    // Generate the random image
    cv::Mat image = generateRandomImage(width, height);

    // Optionally, save the image to a file
    cv::imwrite("random_image.png", image);
    std::cout << "Random image saved to random_image.png" << std::endl;

    // Convert the image to float and normalize to [0, 1]
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F, 1.0 / 255.0);

    int imageHeight = imageFloat.rows;
    int imageWidth = imageFloat.cols;

  typedef float floatTypeA;
  typedef float floatTypeB;
  typedef float floatTypeC;
  typedef float floatTypeCompute;

  cudaDataType_t typeA = CUDA_R_32F;
  cudaDataType_t typeB = CUDA_R_32F;
  cudaDataType_t typeC = CUDA_R_32F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

  floatTypeCompute alpha = (floatTypeCompute)1.0f; // Changed to 1.0 for convolution
  floatTypeCompute beta = (floatTypeCompute)0.f;

  /**********************
   * Image Convolution: C_{m,n} = alpha * A_{m,k} B_{k,n} + beta * C_{m,n}
   * Simplified to: output_image = kernel * input_image
   * Where A is the kernel, B is the image, and C is the output image
   **********************/

  std::vector<int> modeC{'m', 'n'}; // Output image
  std::vector<int> modeA{'m', 'k'}; // Kernel
  std::vector<int> modeB{'k', 'n'}; // Input image

  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();

  std::unordered_map<int, int64_t> extent;
  int kernelSize = 3; // Example: 3x3 kernel
  extent['m'] = kernelSize * kernelSize; // Kernel size (flattened)
  extent['n'] = imageHeight * imageWidth; // Image size (flattened)
  extent['k'] = kernelSize * kernelSize; // Kernel size (flattened)

  double gflops = (2.0 * extent['m'] * extent['n'] * extent['k']) / 1e9;

  std::vector<int64_t> extentC;
  for (auto mode : modeC) extentC.push_back(extent[mode]);
  std::vector<int64_t> extentA;
  for (auto mode : modeA) extentA.push_back(extent[mode]);
  std::vector<int64_t> extentB;
  for (auto mode : modeB) extentB.push_back(extent[mode]);

  /**********************
   * Allocating data
   **********************/

  size_t elementsA = 1;
  for (auto mode : modeA) elementsA *= extent[mode];
  size_t elementsB = 1;
  for (auto mode : modeB) elementsB *= extent[mode];
  size_t elementsC = 1;
  for (auto mode : modeC) elementsC *= extent[mode];

  size_t sizeA = sizeof(floatTypeA) * elementsA;
  size_t sizeB = sizeof(floatTypeB) * elementsB;
  size_t sizeC = sizeof(floatTypeC) * elementsC;
  printf("Total memory: %.2f MiB\n",
         (sizeA + sizeB + sizeC) / 1024. / 1024.);

  void *A_d, *B_d, *C_d;
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeA));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeB));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeC));

  floatTypeA *A = (floatTypeA *)malloc(sizeof(floatTypeA) * elementsA);
  floatTypeB *B = (floatTypeB *)malloc(sizeof(floatTypeB) * elementsB);
  floatTypeC *C = (floatTypeC *)malloc(sizeof(floatTypeC) * elementsC);

  if (A == NULL || B == NULL || C == NULL) {
    printf("Error: Host allocation of A or C.\n");
    return -1;
  }

  /*******************
   * Initialize data
   *******************/

    // Initialize the kernel (example: Gaussian blur)
    float kernel[9] = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };

    for (int64_t i = 0; i < elementsA; i++) {
        A[i] = kernel[i];
    }

    // Flatten the image into a 1D array
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            B[i * imageWidth + j] = imageFloat.at<float>(i, j);
        }
    }

    // Initialize C to zero
    for (int64_t i = 0; i < elementsC; i++) {
        C[i] = 0.0f;
    }

  HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

  /*************************
   * cuTENSOR
   *************************/

  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorInit(&handle));

  /**********************
   * Setup planCache
   **********************/
  constexpr int32_t numCachelines = 1024;
  size_t sizeCache = numCachelines * sizeof(cutensorPlanCacheline_t);
  printf("Allocating: %.2f kB for the cache\n", sizeCache / 1000.);
  cutensorPlanCacheline_t *cachelines =
      (cutensorPlanCacheline_t *)malloc(sizeCache);
  HANDLE_ERROR(cutensorHandleAttachPlanCachelines(&handle, cachelines,
                                                  numCachelines));

  const char cacheFilename[] = "./cache.bin";
  uint32_t numCachelinesRead = 0;
  cutensorStatus_t status = cutensorHandleReadCacheFromFile(
      &handle, cacheFilename, &numCachelinesRead);
  if (status == CUTENSOR_STATUS_SUCCESS) {
    printf(
        "%d cachelines have been successfully read from file (%s).\n",
        numCachelinesRead, cacheFilename);
  } else if (status == CUTENSOR_STATUS_IO_ERROR) {
    printf("File (%s) doesn't seem to exist.\n", cacheFilename);
  } else if (status == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE) {
    printf(
        "Cannot read cache: Please attach at least %d cachelines to the "
        "handle.\n",
        numCachelinesRead);
  }

  /**********************
   * Create Tensor Descriptors
   **********************/

  cutensorTensorDescriptor_t descA;
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descA, nmodeA,
                                            extentA.data(), NULL, /*stride*/
                                            typeA, CUTENSOR_OP_IDENTITY));

  cutensorTensorDescriptor_t descB;
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descB, nmodeB,
                                            extentB.data(), NULL, /*stride*/
                                            typeB, CUTENSOR_OP_IDENTITY));

  cutensorTensorDescriptor_t descC;
  HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &descC, nmodeC,
                                            extentC.data(), NULL, /*stride*/
                                            typeC, CUTENSOR_OP_IDENTITY));

  /**********************************************
   * Retrieve the memory alignment for each tensor
   **********************************************/

  uint32_t alignmentRequirementA;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle, A_d, &descA,
                                               &alignmentRequirementA));

  uint32_t alignmentRequirementB;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle, B_d, &descB,
                                               &alignmentRequirementB));

  uint32_t alignmentRequirementC;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle, C_d, &descC,
                                               &alignmentRequirementC));

  /*******************************
   * Create Contraction Descriptor
   *******************************/

  cutensorContractionDescriptor_t desc;
  HANDLE_ERROR(cutensorInitContractionDescriptor(
      &handle, &desc, &descA, modeA.data(), alignmentRequirementA, &descB,
      modeB.data(), alignmentRequirementB, &descC, modeC.data(),
      alignmentRequirementC, &descC, modeC.data(), alignmentRequirementC,
      typeCompute));

  /**************************
   * Set the algorithm to use
   ***************************/

  cutensorContractionFind_t find;
  HANDLE_ERROR(cutensorInitContractionFind(&handle, &find,
                                           CUTENSOR_ALGO_DEFAULT));

  const cutensorCacheMode_t cacheMode = CUTENSOR_CACHE_MODE_PEDANTIC;
  HANDLE_ERROR(cutensorContractionFindSetAttribute(
      &handle, &find, CUTENSOR_CONTRACTION_FIND_CACHE_MODE, &cacheMode,
      sizeof(cutensorCacheMode_t)));

  const cutensorAutotuneMode_t autotuneMode =
      CUTENSOR_AUTOTUNE_INCREMENTAL;
  HANDLE_ERROR(cutensorContractionFindSetAttribute(
      &handle, &find, CUTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE, &autotuneMode,
      sizeof(cutensorAutotuneMode_t)));

  const uint32_t incCount = 4;
  HANDLE_ERROR(cutensorContractionFindSetAttribute(
      &handle, &find, CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT, &incCount,
      sizeof(uint32_t)));

  /**********************
   * Query workspace
   **********************/

  uint64_t worksize = 0;
  HANDLE_ERROR(cutensorContractionGetWorkspace(&handle, &desc, &find,
                                               CUTENSOR_WORKSPACE_MAX,
                                               &worksize));  // TODO

  void *work = nullptr;
  if (worksize > 0) {
    if (cudaSuccess != cudaMalloc(&work, worksize)) {
      work = nullptr;
      worksize = 0;
    }
  }

  /**************************
   * Create Contraction Plan
   **************************/

  cutensorContractionPlan_t plan;

  /**********************
   * Run
   **********************/

  double minTimeCUTENSOR = 1e100;
  // warm-up GPU (without caching) (optional, but recommended for more
  // accurate measurements later on)
  for (int i = 0; i < 4; ++i) {
    cutensorContractionFind_t find_copy = find;

    const cutensorCacheMode_t cacheMode = CUTENSOR_CACHE_MODE_NONE;
    HANDLE_ERROR(cutensorContractionFindSetAttribute(
        &handle, &find_copy, CUTENSOR_CONTRACTION_FIND_CACHE_MODE, &cacheMode,
        sizeof(cutensorCacheMode_t)));

    // To take advantage of the incremental-autotuning (via the cache), it's
    // important to re-initialize the plan
    HANDLE_ERROR(cutensorInitContractionPlan(&handle, &plan, &desc,
                                             &find_copy, worksize));

    HANDLE_ERROR(cutensorContraction(&handle, &plan, (void *)&alpha, A_d, B_d,
                                     (void *)&beta, C_d, C_d, work, worksize,
                                     0 /* stream */));
  }
  cudaDeviceSynchronize();
  printf("Warm-up completed.\n");

  for (int i = 0; i < incCount + 1; ++i)  // last iteration will hit the cache
  {
    cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Set up timing
    GPUTimer timer;
    timer.start();

    // To take advantage of the incremental-autotuning (via the cache), it's
    // important to re-initialize the plan
    HANDLE_ERROR(cutensorInitContractionPlan(&handle, &plan, &desc, &find,
                                             worksize));

    cutensorStatus_t err = cutensorContraction(
        &handle, &plan, (void *)&alpha, A_d, B_d, (void *)&beta, C_d, C_d, work,
        worksize, 0 /* stream */);

    // Synchronize and measure timing
    auto time = timer.seconds();

    if (err != CUTENSOR_STATUS_SUCCESS) {
      printf("ERROR: %s in %s:%d\n", cutensorGetErrorString(err), __FILE__,
             __LINE__);
      break;
    }
    minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
  }

  /*************************/

  double transferedBytes = sizeC + sizeA + sizeB;
  transferedBytes += ((float)beta != 0.f) ? sizeC : 0;
  transferedBytes /= 1e9;
  printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", gflops / minTimeCUTENSOR,
         transferedBytes / minTimeCUTENSOR);

  /*
   * Optional: Write cache to disk
   */
  HANDLE_ERROR(cutensorHandleWriteCacheToFile(&handle, cacheFilename));
  printf("Cache has been successfully written to file (%s).\n", cacheFilename);

  // Detach cache and free-up resources
  HANDLE_ERROR(cutensorHandleDetachPlanCachelines(&handle));

    // Copy the result back to the host
    HANDLE_CUDA_ERROR(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Reshape the result back into an image
    cv::Mat outputImage(imageHeight, imageWidth, CV_32F);
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            outputImage.at<float>(i, j) = C[i * imageWidth + j];
        }
    }

    // Normalize the output image to [0, 255] and convert to 8-bit
    cv::Mat outputImage8U;
    cv::normalize(outputImage, outputImage8U, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imwrite("random_image_alter.png", outputImage8U);
    std::cout << "alter Random image saved to random_image_alter.png" << std::endl;

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (cachelines) free(cachelines);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

  return 0;
}
