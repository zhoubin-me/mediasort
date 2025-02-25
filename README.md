# Media Sort

A command-line tool for organizing and deduplicating photo/video collections using content-based image similarity detection.
[![asciicast](https://asciinema.org/a/C6IpPeFu5i2i8whGn077otGlS.svg)](https://asciinema.org/a/C6IpPeFu5i2i8whGn077otGlS)

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/bzhouxyz)

## Features

- Recursively scans source directory for photos and videos
- Detects and removes duplicate/similar images using perceptual hashing and DINOv2 neural network features
- Organizes media files by date into year/month folders
- Caches processed file information to speed up subsequent runs
- Supports both photo and video files
- Shows detailed statistics about your media collection
- Interactive confirmation before moving/deleting files

## Installation

Download libtorch (C++ distribution of PyTorch):

For CPU-only:

```bash
curl -L https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip -o libtorch.zip
```

For CUDA:

```bash
curl -L https://download.pytorch.org/libtorch/nightly/cu126/libtorch-shared-with-deps-latest.zip -o libtorch-gpu.zip
```

For AMD GPU:

```bash
curl -L https://download.pytorch.org/libtorch/rocm6.2.4/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Brocm6.2.4.zip -o libtorch-amd.zip
```

Go to [libtorch download page](https://pytorch.org/get-started/locally/) to get the latest version.

Extract the zip file:

```bash
unzip libtorch.zip
mv libtorch /opt/libtorch
```

Export Environment Variable:

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

To organize photo files:

```bash
cargo run --release -- --source /path/to/source --target /path/to/photo_target
```

To organize video files:
```bash
cargo run --release -- --source /path/to/source --target /path/to/video_target --video
```






