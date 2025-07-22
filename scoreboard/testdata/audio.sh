#!/bin/bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

# sudo apt install ffmpeg libttspico-utils

if [ ! -f audio.wav ]; then
  pico2wave -w audio.wav "orange"
fi

if [ ! -f audio.aac ]; then
  ffmpeg -hide_banner -loglevel error -i audio.wav -c:a aac -b:a 24k audio.aac
fi

if [ ! -f audio.flac ]; then
  ffmpeg -hide_banner -loglevel error -i audio.wav -ac 1 -c:a flac -compression_level 12 audio.flac
fi

if [ ! -f audio.mp3 ]; then
  ffmpeg -hide_banner -loglevel error -i audio.wav -ac 1 -b:a 32k audio.mp3
fi

if [ ! -f audio.ogg ]; then
  ffmpeg -hide_banner -loglevel error -i audio.wav -c:a libopus -b:a 24k audio.ogg
fi
