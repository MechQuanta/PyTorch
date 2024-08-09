import torch
import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
"""print(torch.cuda.is_available())
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/CPU:0'):
  c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  d = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
c = tf.matmul(c, d)
print(c)"""

"""gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)"""

"""The code checks for available GPUs.
It restricts TensorFlow to use only the first GPU.
It then prints the number of physical and logical GPUs.
If there's an error (like trying to set visible devices after initialization), it handles it gracefully by printing the error."""


"""gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)"""


"""gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""
"""print(np.finfo(float).eps)"""
"""x = torch.tensor([[1,2],[3,4]])
y = torch.tensor([[5,6],[7,8]])
z = torch.hstack([x,y])
print(z)
data = scipy.io.loadmat("./PINNs/Cylinder_wake/Cylinder_wake.mat")"""
"""x = np.array([[1,3],[2,4]])
y = x.flatten()[:,None]
print(y)"""
"""print(torch.version.cuda)
print(torch.cuda.is_available())
print("Current device : ", torch.cuda.current_device())
print("Device name : ", torch.cuda.get_device_name(0))"""
"""x = np.ones((5000,1))
print(x)"""



"""# Create some example data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create a filled contour plot
plt.contourf(X, Y, Z, levels=30, cmap='jet')

# Add a color bar to the plot
plt.colorbar()

plt.show()
"""

"""x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
X,Y = np.meshgrid(x,y)
print("X = ",X)
print("Y = ",Y)
print(X.shape)
print(Y.shape)"""


"""import numpy as np
import matplotlib.pyplot as plt
"""
"""# Create 1D arrays
x = np.linspace(-3, 3, 100)  # 100 points between -3 and 3
y = np.linspace(-3, 3, 100)

# Create 2D grid using meshgrid
X, Y = np.meshgrid(x, y)

# Evaluate a function over the grid
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create a filled contour plot
plt.contourf(X, Y, Z, levels=30, cmap='jet')

# Add a color bar to indicate the scale
plt.colorbar()

# Show the plot
plt.show()
"""
x = np.array([[1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
              [10]])
y = np.reshape(x,(5,2))
print("x = ",x)
print("y = ",y)
