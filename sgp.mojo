from pathlib import Path
from python import Python
from python import PythonObject

fn read_jpg_images_from_directory(directory_path: String) raises:
    """Read all JPG images from a directory using Python's PIL library."""
    
    print("\n=== JPG Image Reader ===")
    print("Attempting to read from directory:", directory_path)
    
    # Import Python modules
    var os = Python.import_module("os")
    
    # Get list of files in directory
    var files = os.listdir(directory_path)
    print("Found", len(files), "files in directory")
    
    # Try to import PIL with error handling
    try:
        var Image = Python.import_module("PIL.Image")
        print("PIL.Image imported successfully")
        
        print("Importing numpy...")
        var np = Python.import_module("numpy")
        print("Numpy imported successfully")
        
        # Process first few JPG files as a test
        var jpg_count = 0
        var max_files_to_process = 2  # Process only first 2 files for testing
        
        for i in range(len(files)):
            if jpg_count >= max_files_to_process:
                break
                
            var filename = files[i]
            var filename_str = String(filename)
            
            # Check if file is a JPG (case insensitive)
            var lower_filename = filename_str.lower()
            var is_jpg = lower_filename.endswith(".jpg") or lower_filename.endswith(".jpeg")
            
            if is_jpg:
                jpg_count += 1
                var full_path = directory_path + "/" + filename_str
                print("\nProcessing JPG #", jpg_count, ":", filename_str)
                
                try:
                    print("  Opening image...")
                    var img = Image.open(full_path)
                    
                    print("  Getting image info...")
                    var width = img.width
                    var height = img.height
                    var mode = img.mode
                    
                    print("  Dimensions:", width, "x", height)
                    print("  Mode:", mode)
                    
                    print("  Converting to numpy array...")
                    var img_array = np.array(img)
                    var shape = img_array.shape
                    print("  Array shape:", shape)
                    
                    print("  Closing image...")
                    img.close()
                    print("  ✓ Successfully processed:", filename_str)
                    
                except:
                    print("  ✗ Error processing image:", filename_str)
        
        if jpg_count == 0:
            print("No JPG files found in directory")
        else:
            print("\nProcessed", jpg_count, "JPG files")
        
    except:
        print("✗ Error importing PIL or numpy")
        return

from python import Python

fn use_glfw() raises:
    var glfw = Python.import_module("glfw")
    #var vk = Python.import_module("vulkan")

    # Initialize GLFW with Vulkan hints
    #glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # Required for Vulkan
    #glfw.window_hint(glfw.RESIZABLE, False)  # Simplifies Vulkan setup

    if not glfw.init():
        raise Error("GLFW initialization failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create window
    var window = glfw.create_window(800, 600, "Vulkan + Wayland (Ctrl+Q to quit)", None, None)
    if not window:
        glfw.terminate()
        raise Error("Window creation failed")

    glfw.make_context_current(window)

    # Basic Vulkan setup
    #var instance_info = vk.VkInstanceCreateInfo(
    #    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    #    pApplicationInfo=vk.VkApplicationInfo(
    #        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    #        pApplicationName="Vulkan Mojo",
    #        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    #        pEngineName="No Engine",
    #        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    #        apiVersion=vk.VK_API_VERSION_1_0
    #    )
    #)

    #var instance = vk.VkInstance(0)
    #if vk.vkCreateInstance(instance_info, None, instance) != vk.VK_SUCCESS:
    #    glfw.terminate()
    #    raise Error("Vulkan instance creation failed")

    # Main loop
    while not glfw.window_should_close(window):
        # Check for Ctrl+Q
        #if (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS and
        #   glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS):
        #    glfw.set_window_should_close(window, True)
        glfw.poll_events()
        glfw.swap_buffers(window)

    # Cleanup
    #vk.vkDestroyInstance(instance, None)
    glfw.terminate()

fn main():
    """Main function to demonstrate reading JPG images."""
    
    # Then try to read images
    var image_directory = "./data/images"
    
    try:
        read_jpg_images_from_directory(image_directory)
        print("\n=== Completed successfully ===")
    except:
        print("\n=== Error occurred during execution ===")
        
        try:
            var os = Python.import_module("os")
            var cwd = os.getcwd()
            print("Current working directory:", cwd)
        except:
            print("Could not get current directory")

    try:
        use_glfw()
    except e:
        print("\n=== Error using glfw: ", e)
