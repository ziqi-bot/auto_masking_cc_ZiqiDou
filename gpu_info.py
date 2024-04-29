import tensorflow as tf

# Get detailed information about all GPU devices
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("Device Name:", gpu.name)
        print("Device Type:", gpu.device_type)
        # Get the memory limit of the GPU device
        device_memory = tf.test.gpu_device_name()  # Get GPU device name
        if device_memory:
            print("Device Memory Limit:", f"{tf.test.gpu_device_name()} ({tf.test.gpu_device_name()})")
        else:
            print("No memory information found for the GPU device.")
        print()
else:
    print("No available GPU devices.")

# Alternatively, use tf.test.is_gpu_available() to check for GPU existence
if tf.test.is_gpu_available():
    print("There are available GPU devices.")
else:
    print("No available GPU devices.")

# Print the TensorFlow version
print("TensorFlow version:", tf.__version__)
