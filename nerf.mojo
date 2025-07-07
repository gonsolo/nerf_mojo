from python import Python
from gpu.host import DeviceContext

def main():
        context = DeviceContext()
        np = Python.import_module("numpy")
        plt = Python.import_module("matplotlib.pyplot")
        data = np.load('tiny_nerf_data.npz')
        images = data['images']
        poses = data['poses']
        focal = data['focal']

        print('Images shape: ', images.shape)
        print('Poses shape: ', poses.shape)
        print('Focal length: ', focal)

        height = images.shape[1]
        width = images.shape[2]
        near = 2.0
        far = 6.0
        n_training = 100
        testimg_idx = 101
        testimg = images[testimg_idx]
        testpose = poses[testimg_idx]
        plt.imshow(testimg)
        plt.show()
        print('Pose')
        print(testpose)
