from gpu.host import DeviceContext
from python import Python

def load_data(path: String) -> (List[List[List[List[Float64]]]], List[List[List[Float64]]], Float64):
    var np = Python.import_module("numpy")
    var data = np.load(path)
    var py_images = data["images"]
    var py_poses = data["poses"]
    var py_focal = data["focal"]

    # Convert images
    var images = List[List[List[List[Float64]]]]()
    for img in py_images:
        var img_list = List[List[List[Float64]]]()
        for row in img:
            var row_list = List[List[Float64]]()
            for pixel in row:
                var pixel_list = List[Float64]()
                pixel_list.append(Float64(pixel[0]))
                pixel_list.append(Float64(pixel[1]))
                pixel_list.append(Float64(pixel[2]))
                row_list.append(pixel_list)
            img_list.append(row_list)
        images.append(img_list)

    # Convert poses
    var poses = List[List[List[Float64]]]()
    for py_pose in py_poses:
        var pose = List[List[Float64]]()
        for i in range(4):
            var row = List[Float64]()
            for j in range(4):
                row.append(Float64(py_pose[i][j]))
            pose.append(row)
        poses.append(pose)

    # Convert focal
    var focal = Float64(py_focal.item())
    return images, poses, focal

def main():
    try:
        context = DeviceContext()
        var plt = Python.import_module("matplotlib.pyplot")
        var np = Python.import_module("numpy")

        var result = load_data("tiny_nerf_data.npz")
        var images = result[0]
        var poses = result[1]
        var focal = result[2]

        var height = len(images[0])
        var width = len(images[0][0])
        print("Images shape:", len(images), height, width, len(images[0][0][0]))
        print("Poses shape:", len(poses), 4, 4)
        print("Focal length:", focal)

        # Extract origins and dirs
        var origins = List[List[Float64]]()
        var dirs = List[List[Float64]]()
        for pose in poses:
            var origin = List[Float64]()
            origin.append(pose[0][3])
            origin.append(pose[1][3])
            origin.append(pose[2][3])
            origins.append(origin)

            # Calculate direction vector: [0, 0, -1] * pose[:3, :3]
            # This is equivalent to taking the negative third column of the rotation matrix
            var dir_vec = List[Float64]()
            dir_vec.append(-pose[0][2])  # -pose[0, 2]
            dir_vec.append(-pose[1][2])  # -pose[1, 2]
            dir_vec.append(-pose[2][2])  # -pose[2, 2]
            dirs.append(dir_vec)
        print("First origin:", origins[0][0], origins[0][1], origins[0][2])
        print("First dir sum:", dirs[0][0], dirs[0][1], dirs[0][2])

        # Convert Mojo Lists to Python lists manually
        var py_origins = Python.evaluate("[]")
        for origin in origins:
            var py_origin = Python.evaluate("[]")
            for coord in origin:
                py_origin.append(coord)
            py_origins.append(py_origin)

        var py_dirs = Python.evaluate("[]")
        for dir_vec in dirs:
            var py_dir = Python.evaluate("[]")
            for coord in dir_vec:
                py_dir.append(coord)
            py_dirs.append(py_dir)

        # Convert to numpy arrays
        var origins_np = np.array(py_origins)
        print(origins_np.dtype)
        print(origins_np.shape)
        print(origins_np[0])

        var dirs_np = np.array(py_dirs)

        _ = context # TODO: Remove this when context is used
    except:
        print("Error: Failed to load data")
        return
