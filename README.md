# video-sampler

<div align="center">

[![Python Version](https://img.shields.io/pypi/pyversions/video-sampler.svg)](https://pypi.org/project/video-sampler/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/LemurPwned/video-sampler/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/LemurPwned/video-sampler/blob/main/.pre-commit-config.yaml)

[![License](https://img.shields.io/github/license/LemurPwned/video-sampler)](https://github.com/LemurPwned/video-sampler/blob/main/LICENSE)

Video sampler allows you to efficiently sample video frames.
Currently it uses keyframe decoding, frame interval gating and perceptual hashing to reduce duplicated samples.

**Use case:** for sampling videos for later annotations used in machine learning.

</div>

## Installation and Usage

```bash
pip install -U video-sampler
```

then you can run

```bash
python3 -m video-sampler --help
```

### Basic usage

```bash
python3 -m  video-sampler FatCat.mp4 ./dataset-frames/ --hash-size 3 --buffer-size 20
```

## Benchmarks

Configuration for this benchmark:

```bash
SamplerConfig(min_frame_interval_sec=1.0, keyframes_only=True, buffer_size=10, hash_size=X, queue_wait=0.1, debug=True)
```

|                                    Video                                    | Hash size | Decoded | Saved |
| :-------------------------------------------------------------------------: | :-------: | :-----: | :---: |
| [Fat Cat Video](https://www.youtube.com/watch?v=kgrV3_g9rYY&ab_channel=BBC) |     8     |   297   |  278  |
| [Fat Cat Video](https://www.youtube.com/watch?v=kgrV3_g9rYY&ab_channel=BBC) |     4     |   297   |  173  |
|           [SmolCat](https://www.youtube.com/watch?v=W86cTIoMv2U)            |     8     |   118   |  106  |
|           [SmolCat](https://www.youtube.com/watch?v=W86cTIoMv2U)            |     4     |   118   |  62   |
|          [HighLemurs](https://www.youtube.com/watch?v=yYXoCHLqr4o)          |     8     |   458   |  441  |
|          [HighLemurs](https://www.youtube.com/watch?v=yYXoCHLqr4o)          |     4     |   458   |  309  |

## Flit commands

#### Build

```
flit build
```

#### Install

```
flit install
```

#### Publish

```
flit publish
```

## 🛡 License

[![License](https://img.shields.io/github/license/LemurPwned/video-sampler)](https://github.com/LemurPwned/video-sampler/blob/main/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/LemurPwned/video-sampler/blob/main/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{video-sampler,
  author = {video-sampler},
  title = {Video sampler allows you to efficiently sample video frames},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LemurPwned/video-sampler}}
}
```
