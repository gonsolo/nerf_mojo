from pathlib import Path
from python import Python

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

fn use_glfw() raises:
    var glfw = Python.import_module("glfw")

    # Initialize GLFW
    if not glfw.init():
        raise Error("Failed to initialize GLFW")

    # Configure window hints
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create a window
    var window = glfw.create_window(640, 480, "Mojo + GLFW", None, None)
    if not window:
        glfw.terminate()
        raise Error("Failed to create GLFW window")

    # Make the window's context current
    glfw.make_context_current(window)

    # Main loop
    while not glfw.window_should_close(window):
        # Render here
        glfw.swap_buffers(window)
        glfw.poll_events()

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
    except:
        print("\n=== Error using glfw ===")
