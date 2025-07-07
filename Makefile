#all: python
all: mojo
mojo:
	mojo nerf.mojo
python:
	python nerf.py
edit:
	vi -O nerf.mojo nerf.py
