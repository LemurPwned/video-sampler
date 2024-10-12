# Changelog

## Description

Changelog for the `video-sampler`.

### 0.11.1 - 0.11.2

- added `start-time` and `end-time` options to the `video-sampler` command which allows to specify the start and end time of the video segment to be processed.
- fixed optional imports
- minimal docker builds added

### 0.11.0

- added multiprocessing over multiple videos in the folder
- new cli command & support reading from `.yaml` config files

### 0.10.0

- added summary creation from sampled frames

### 0.9.0

- keyword yt-dlp extraction

### 0.8.0

- added yt-dlp integration
- extra yt-dlp options are supported

### 0.7.0

- added blur gating

### 0.6.0

- adding CLIP-based gating for sampling
- added docker build image
- fixed some bugs in requirements and setup scripts
- adding gif creation from sampled frames
- added a notebook in `notebooks` to show the distribution shift of the sampled frames
- added some basic tests (really basic)
- added a short explanation and usage of guided sampling in README.md.
- added grid hash sampling

### 0.5.0

- fixed a major bug in the hashing buffer
- added entropy and gzip methods for the sliding top-k buffer
- improved build process
