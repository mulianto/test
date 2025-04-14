README.txt

Project: Image Processing with NVIDIA Performance Primitives (NPP)

Description:
------------
This project demonstrates image processing using the NVIDIA Performance Primitives (NPP) library. The code, `boxFilterNPP.cpp`, performs the following operations on a grayscale image:

1.  **Image Loading:** Loads a grayscale image (PGM format) from disk.
2.  **Gaussian Blur:** Applies a Gaussian blur to the image to reduce noise and smooth it.
3.  **Sobel Edge Detection:** Performs Sobel edge detection (both vertical and horizontal) on the blurred image.
4. **Save the result:** Saves the blurred image to a new file.

Note: The Sobel result is calculated, but not used or saved. The thresholding step is completely commented out.

File:
-----
*   `boxFilterNPP.cpp`: The main C++ source code file.

Prerequisites:
--------------
*   **CUDA Toolkit:** The NVIDIA CUDA Toolkit must be installed. This includes the NPP library.
*   **C++ Compiler:** A C++ compiler (e.g., g++) that is compatible with the CUDA Toolkit.
*   **CUDA SDK:** The CUDA SDK (or a similar source) must be installed to provide the following header files:
    *   `Exceptions.h`
    *   `ImageIO.h`
    *   `ImagesCPU.h`
    *   `ImagesNPP.h`
    *   `helper_cuda.h`
    *   `helper_string.h`
* **PGM image:** A PGM image to process.

Compilation:
------------
1.  **Navigate:** Open a terminal or command prompt and navigate to the directory containing `boxFilterNPP.cpp` and the required header files.
2.  **Compile:** Use the `nvcc` compiler (part of the CUDA Toolkit) to compile the code:
    ```bash
    nvcc -o boxFilterNPP boxFilterNPP.cpp -lnppc -lnppi -lnpps -lcudart
    ```
    *   `-o boxFilterNPP`: Specifies the output executable name.
    *   `boxFilterNPP.cpp`: The source code file.
    *   `-lnppc`: Links the NPP Core library.
    *   `-lnppi`: Links the NPP Image Processing library.
    *   `-lnpps`: Links the NPP Signal Processing library.
    *   `-lcudart`: Links the CUDA runtime library.
    * `-I../Common`: Add the path to the common header files.
    * `-I../Common/UtilNPP`: Add the path to the UtilNPP header files.

Usage:
------
1.  **Run:** Execute the compiled program:
    ```bash
    ./boxFilterNPP
    ```
    *   This will process the default image `Lena.pgm` (if it exists in the same directory or can be found using `sdkFindFilePath`) and save the blurred image as `Lena_processed.pgm`.

2.  **Input Image:** To specify a different input image, use the `-input` command-line argument:
    ```bash
    ./boxFilterNPP -input=<your_image.pgm>
    ```
    *   Replace `<your_image.pgm>` with the actual path to your PGM image.

3.  **Output Image:** To specify a different output image name, use the `-output` command-line argument:
    ```bash
    ./boxFilterNPP -output=<your_output_image.pgm>
    ```
    *   Replace `<your_output_image.pgm>` with the desired output filename.

4.  **Both Input and Output:** To specify both the input and output images:
    ```bash
    ./boxFilterNPP -input=<your_image.pgm> -output=<your_output_image.pgm>
    ```

Notes:
------
*   The code assumes the input image is in PGM (Portable Gray Map) format.
*   The output image will also be in PGM format.
*   The code includes error handling for file I/O and NPP function calls.
*   The code prints information about the NPP library and CUDA versions.
* The code only outputs the blurred image.
* The Sobel result is calculated, but not used or saved.
* The thresholding step is completely commented out.

