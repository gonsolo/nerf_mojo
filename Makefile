all: view
mojo:
	mojo nerf.mojo
view:
	python view.py
train:
	python train.py
edit:
	vi view.py
