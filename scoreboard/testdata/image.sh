#!/bin/bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

# sudo apt install curl ffmpeg pngcrush

# Copyright Evan-Amos.
# Source: https://en.m.wikipedia.org/wiki/File:Banana-Single.jpg

if [ ! -f image.jpg ]; then
	curl -sSL -o image.jpg 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/330px-Banana-Single.jpg'
fi

if [ ! -f image.gif ]; then
  ffmpeg -hide_banner -loglevel error -i image.jpg -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" image.gif
fi

# TODO: I need to update my old OS or install newer ffmpeg.
#if [ ! -f image.heic ]; then
#  ffmpeg -hide_banner -loglevel error -i image.jpg -c:v libx265 -crf 40 image.heic
#fi

if [ ! -f image.png ]; then
  ffmpeg -hide_banner -loglevel error -i image.jpg image.png
  pngcrush -brute image.png image2.png
  mv image2.png image.png
fi

if [ ! -f image.webp ]; then
  ffmpeg -hide_banner -loglevel error -i image.jpg image.webp
fi
