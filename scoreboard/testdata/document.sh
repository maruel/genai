#!/bin/bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

# sudo apt install texlive-latex-base

if [ ! -f document.pdf ]; then
	pdflatex document.tex
	rm document.aux document.log
fi
