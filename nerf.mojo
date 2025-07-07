from python import Python
from gpu.host import DeviceContext

def main():
        var context = DeviceContext()
        var np = Python.import_module("numpy")
        var data = np.load('tiny_nerf_data.npz')
        var images = data['images']
        var poses = data['poses']
        var focal = data['focal']

        print('Images shape: ', images.shape)
        print('Poses shape: ', poses.shape)
        print('Focal length: ', focal)
