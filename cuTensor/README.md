# cuTENSOR Image Convolution Example

This project demonstrates how to perform image convolution using NVIDIA's cuTENSOR library. It leverages the power of GPUs to accelerate the convolution operation, which is a fundamental building block in many image processing and computer vision tasks.

## Overview

The code performs a 2D convolution of a randomly generated grayscale image with a 3x3 Gaussian blur kernel. It utilizes the cuTENSOR library for efficient tensor contractions on NVIDIA GPUs. The project also incorporates OpenCV for image generation, loading, saving, and basic image manipulation.

## Features

*   **GPU-Accelerated Convolution:** Utilizes cuTENSOR for high-performance tensor contractions on NVIDIA GPUs.
*   **Random Image Generation:** Generates a random grayscale image using OpenCV.
*   **Gaussian Blur Kernel:** Implements a 3x3 Gaussian blur kernel for the convolution operation.
*   **Image Normalization:** Normalizes the input image to the [0, 1] range and the output image to [0, 255].
*   **Caching:** Demonstrates cuTENSOR's caching mechanism for performance optimization.
*   **Performance Measurement:** Measures and reports the GFLOPs/s and GB/s achieved by the cuTENSOR convolution.
* **OpenCV Integration:** Uses OpenCV for image creation, manipulation, and saving.

## Dependencies

*   **CUDA Toolkit:** Required for GPU programming and cuTENSOR.
*   **cuTENSOR Library:** NVIDIA's library for high-performance tensor contractions.
*   **OpenCV:** For image processing tasks (image generation, loading, saving, and manipulation).
*   **C++ Compiler:** A C++ compiler that supports C++11 or later (e.g., g++, clang++).
* **CMake:** For building the project.

## Installation and Build Instructions

1.  **Install Dependencies:**
    *   **CUDA Toolkit:** Download and install the CUDA Toolkit from the NVIDIA website.
    *   **cuTENSOR:** The cuTENSOR library is typically included with the CUDA Toolkit. Ensure it's properly installed.
    *   **OpenCV:** Install OpenCV using your system's package manager (e.g., `apt-get install libopencv-dev` on Ubuntu) or by building it from source.
    * **CMake:** Install CMake using your system's package manager.

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/mulianto/test 
    cd cuTensor
    ```

3.  **Build the Project:**
    look at the MakeFile

4. **Run the executable**
    ```bash
    ./cutensor_example.exe
    ```
    * The executable will be located in the build directory.

## Running the Code

After building the project, you can run the executable. The program will:

1.  Generate a random grayscale image (512x512 pixels).
2.  Save the random image as `random_image.png`.
3.  Perform a convolution of the image with a 3x3 Gaussian blur kernel using cuTENSOR.
4.  Save the convolved image as `random_image_alter.png`.
5.  Print the performance metrics (GFLOPs/s and GB/s) to the console.
6. Create a cache file named `cache.bin`

## Code Structure

*   **`cutensor_example.cu`:** The main source file containing the C++ code for image convolution using cuTENSOR and OpenCV.
*   **`CMakeLists.txt`:** (You'll need to create this file) The CMake configuration file for building the project.

## Key Concepts

*   **cuTENSOR:** A library for high-performance tensor contractions on NVIDIA GPUs. It provides functions for initializing tensor descriptors, contraction descriptors, and plans, as well as for performing the actual contraction.
*   **Tensor Contraction:** A generalization of matrix multiplication to higher-order tensors. In this case, it's used to perform the convolution operation.
*   **OpenCV:** A library for computer vision and image processing. It's used here for image generation, loading, saving, and basic manipulation.
*   **Convolution:** A mathematical operation that combines two functions to produce a third function. In image processing, it's used to apply a filter (kernel) to an image.
* **Caching:** cuTENSOR's caching mechanism allows for the storage and reuse of optimized contraction plans, significantly improving performance for repeated operations.

## Example Output

