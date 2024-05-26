FROM python:3.11

# let's copy all the necessary files
# we install our only 2 dependencies :) and vim for nice workflow
RUN apt-get update && \
    apt-get install -y vim
RUN python3 -m pip install --upgrade pip
WORKDIR /scratch
RUN git clone https://github.com/LemurPwned/video-sampler.git && \
    cd video-sampler && \
    python3 -m pip install .[all] && \
    python3 -m spacy download en_core_web_sm
WORKDIR /app
RUN video_sampler --help
