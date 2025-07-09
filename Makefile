all: view
mojo:
	mojo nerf.mojo
python:
	python nerf.py
view:
	python view.py
edit:
	vi -O nerf.mojo nerf.py
