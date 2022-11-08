# AdaNeRF Real-Time TensorRT Viewer Prototype

### [Project Page](https://thomasneff.github.io/adanerf/) | [Video](https://youtu.be/R9rb8tHjMSo) | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6513_ECCV_2022_paper.php) | [DONeRF Dataset](https://repository.tugraz.at/records/jjs3x-4f133)

This directory contains the source code for the real-time viewer prototype used in the AdaNeRF paper. 
Note that this is a research prototype, and as such likely contains bugs and/or warnings.
It was tested on ArchLinux, TensorRT 8.4.1.5 and CUDA 11.7, but it should also work fine on other Unix distributions and Windows.

# Getting Started

1) Make sure that you follow the TensorRT, CUDA and CUDNN installation instructions: 

TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

CUDA: https://docs.nvidia.com/cuda/index.html

CUDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

2) Use an up-to-date version of CMake to generate your project files. 

3) Build the `adanerf` project - this will subsequently also build the dependencies and CUDA kernels in the `adanerf_gpu` project.

4) a.) If the build was successful, you can start the application with the provided sample data as follows (adjust the paths to the sample directory accordingly):

```bash
./adanerf ../../sample/ -ws 800 800 -s 800 800 -bs 80000
```
`-ws` is the window resolution, `-s` is the internal rendering resolution, and `-bs` is the batch size. `src/main.cpp` contains additional listings for command line arguments.

Note: During initial setup for each scene, TensorRT will take a long time building the execution engines for the sampling network and the shading network. For the 4 sample Barbershop in the ´sample/´ directory, this can take somewhere between 10 - 30 minutes. For the 16 sample Pavillon in the ´sample_pavillon_16/´ directory, this can take up to multiple hours.

4) b.) If the build was unsuccessful, make sure that all the prerequisites (TensorRT, CUDA, CUDNN) were found correctly by CMake, as this is the most common issue you can encounter.

Otherwise, this setup will reproduce the results from the AdaNeRF paper in terms of rendering performance, depending on the scene, average sample count, GPU and TensorRT/CUDA version.

# Exporting Networks for Real-Time Rendering

You can use `export.py` from the [root of the AdaNeRF repository](https://github.com/thomasneff/AdaNeRF) to export your trained AdaNeRF models. 
The real-time viewer requires a directory containing the following:

1.) A `config.ini` file that describes hyperparameter settings for the trained network.
2.) A `dataset_info.txt` containing information such as the view cell center and size, depth range and FOV.
3.) Both the depth oracle `.onnx` model and the shading network `.onnx` model, which should be called `model0.onnx` and `model1.onnx` respectively.


# Citation

If you find this repository useful in any way or use/modify AdaNeRF in your research, please consider citing our paper:

```bibtex
@article{kurz-adanerf2022,
 title = {AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields},
 author = {Kurz, Andreas and Neff, Thomas and Lv, Zhaoyang and Zollh\"{o}fer, Michael and Steinberger, Markus},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2022},
}
```
