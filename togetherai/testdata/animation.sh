#!/bin/bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

ffmpeg -hide_banner -y -loglevel error -f lavfi \
	-i color=c=black:s=640x360:duration=4:rate=5 \
	-vf "drawtext=text='BANANA':fontsize=80:fontcolor=yellow:x=(w-text_w)/2:y=(h-text_h)/2:shadowcolor=gray:shadowx=2:shadowy=2,rotate='t':c=black:ow=640:oh=360" \
	-c:v libx264 -crf 40 -preset medium -tune film -pix_fmt yuv420p -an animation.mp4
