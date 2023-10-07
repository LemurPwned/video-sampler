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

or simply

```bash
video-sampler --help
```

### Basic usage

```bash
python3 -m video-sampler hash FatCat.mp4 ./dataset-frames/ --hash-size 3 --buffer-size 20
```

### Advanced usage

There are currently 3 sampling methods:

- `hash` - uses perceptual hashing to reduce duplicated samples
- `entropy` - uses entropy to reduce duplicated samples (work in progress)
- `gzip` - uses gzip compressed size to reduce duplicated samples (work in progress)

To launch either you can run

```bash
video_sampler buffer `method-name` ...other options
```

e.g.

```bash
video_sampler buffer entropy --buffer-size 20 ...
```

where `buffer-size` for `entropy` and `gzip` mean the top-k sliding buffer size. Sliding buffer also uses hashing to reduce duplicated samples.

## Benchmarks

Configuration for this benchmark:

```bash
SamplerConfig(min_frame_interval_sec=1.0, keyframes_only=True, buffer_size=30, hash_size=X, queue_wait=0.1, debug=True)
```

|                                 Video                                 | Total frames | Hash size | Decoded | Saved |
| :-------------------------------------------------------------------: | :----------: | :-------: | :-----: | :---: |
|        [SmolCat](https://www.youtube.com/watch?v=W86cTIoMv2U)         |     2936     |     8     |   118   |  106  |
|        [SmolCat](https://www.youtube.com/watch?v=W86cTIoMv2U)         |      -       |     4     |    -    |  61   |
| [Fat Cat](https://www.youtube.com/watch?v=kgrV3_g9rYY&ab_channel=BBC) |     4462     |     8     |   179   |  163  |
| [Fat Cat](https://www.youtube.com/watch?v=kgrV3_g9rYY&ab_channel=BBC) |      -       |     4     |    -    |  101  |
|       [HighLemurs](https://www.youtube.com/watch?v=yYXoCHLqr4o)       |     4020     |     8     |   161   |  154  |
|       [HighLemurs](https://www.youtube.com/watch?v=yYXoCHLqr4o)       |      -       |     4     |    -    |  126  |

---

```bash
SamplerConfig(
    min_frame_interval_sec=1.0,
    keyframes_only=True,
    queue_wait=0.1,
    debug=False,
    print_stats=True,
    buffer_config={'type': 'entropy'/'gzip', 'size': 30, 'debug': False, 'hash_size': 8, 'expiry': 50}
)
```

|                                 Video                                 | Total frames |  Type   | Decoded | Saved |
| :-------------------------------------------------------------------: | :----------: | :-----: | :-----: | :---: |
|        [SmolCat](https://www.youtube.com/watch?v=W86cTIoMv2U)         |     2936     | entropy |   118   |  39   |
|        [SmolCat](https://www.youtube.com/watch?v=W86cTIoMv2U)         |      -       |  gzip   |    -    |  39   |
| [Fat Cat](https://www.youtube.com/watch?v=kgrV3_g9rYY&ab_channel=BBC) |     4462     | entropy |   179   |  64   |
| [Fat Cat](https://www.youtube.com/watch?v=kgrV3_g9rYY&ab_channel=BBC) |      -       |  gzip   |    -    |  73   |
|       [HighLemurs](https://www.youtube.com/watch?v=yYXoCHLqr4o)       |     4020     | entropy |   161   |  59   |
|       [HighLemurs](https://www.youtube.com/watch?v=yYXoCHLqr4o)       |      -       |  gzip   |    -    |  63   |

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
