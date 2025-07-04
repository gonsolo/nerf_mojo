from python import Python

def main():
        var np = Python.import_module("numpy")
        var data = np.load('tiny_nerf_data.npz')

