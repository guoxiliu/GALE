# GALE

GALE stands for **G**PU-**A**ided **L**ocalized Data Structur**E**, which uses GPU to accelerate the topology-based data analysis algorithm on unstructured meshes (currently supporting triangle and tetrahedral meshes).

## How to Build

We provided detailed steps to build GALE on a Linux machine (tested with Ubuntu 22.04 LTS).

### Prerequisites

To successfully build GALE, please make sure that you have the following libraries installed :  
- [VTK](https://vtk.org/download/) (>= 8.90.0)
- [boost](https://www.boost.org/) (>= 1.74.0)

You also need to have the following build tools installed: 
- [g++](https://gcc.gnu.org/) (>= 9.4.0)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (>= 11.8)
- [CMake](https://cmake.org/) (>= 3.12.0)
- [Ninja](https://ninja-build.org/) (>= 1.10.0) [optional if you want to use Makefile]

### Steps

1. Create the `build` folder in the project root, and change the current working directory to the newly created folder. You can run the following commands in the terminal.

```
mkdir build
cd build
```

2. Run `cmake-gui ../` command in the terminal, and the configuration window shows up. Click on the "Configure" button and select the "Ninja" generator. Once the configuration is finished, make sure `CMAKE_BUILD_TYPE` is set to `Release`. 

3. Once the configuration shows no error, click on the "Generate" button and close the CMake window when the generation process is completed.

4. Run `ninja` command in the terminal to start the compilation process. All the executable files will be created in the `bin` folder.

## How to Run

GALE uses VTK to process the input file, and thus it should support major vtk file types. It also requires that the input dataset contains a scalar field that defines a subdivision of the mesh based on the vertex.

The executable under the `bin` folder accepts the following parameters: 
- `-i <input file>`: the input dataset to process
- `-a <input array>`: the input scalar array included in the dataset
- `-b <block num>`: the number of GPU blocks for the kernel
- `-s <block size>`: the number of threads for each GPU block
- `-p <GPU threads per segment>`: the number of GPU threads that work on one mesh segment
- `-t <CPU thread num>`: the number of CPU threads (for OpenMP)

For example, the following command computes the critical points of the `field` array with `input.vtu` file using 12 CPU threads. The GPU kernels launches with 12 threads blocks and each block contains 128 threads, while 32 GPU threads will be used to compute relation array for one segment.

```
./bin/ttkScalarFieldCriticalPoints -i input.vtu -a field -t 12 -b 12 -s 128 -p 32
```

## License

Please check out [LICENSE](./LICENSE) file for details.
