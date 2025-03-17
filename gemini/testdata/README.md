# Test data

- `animation.mp4` was created with `animation.sh`
- `banana.jpg` was created with GIMP from a random image on the internet (sorry I
  forget where).
- `hidden_word.pdf` was created with `pdflatex hidden_word.tex`
- `mystery_word.opus` was created with
  `espeak -v en-us -p 50 -s 150 "orange" --stdout | ffmpeg -hide_banner -i - -c:a libopus -b:a 24k mystery_word.opus`
