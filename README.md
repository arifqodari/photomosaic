# photomosaic

It is a simple tool that turns your image collection into a mosaic image.

<div>
<img src="https://dl.dropboxusercontent.com/u/16018128/input.jpg"
alt="input image" width=400px></img>
<img src="https://dl.dropboxusercontent.com/u/16018128/output.jpg"
alt="output image" width=400px></img>
</div>

## Dependencies

- OpenCV 2.4.12
- Boost 1.48
- CMake

## Installation

```
mkdir build && cd build
cmake ..
make
```

## Usage

```
./demo -i input.jpg -d image_collection/ -o output.jpg
```
