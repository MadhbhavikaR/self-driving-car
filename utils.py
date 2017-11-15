import cv2
import os
import numpy as numpy
import matplotlib.image as mpimg


IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200 
IMAGE_CHANNELS = 3


def load_image(data_dir, image_file):
  return mpimg.imread(
    os.path.join(
      data_dir, image_file.strip()
    )
  )

def crop_image(image):
  """ Removes sky at the top and car front at the bottom of the image since
      they don't help the model train better.
  """
  return image[60:-25, :, :]

def resize_image(image):
  """ Resizes the image to be the appropriate shape to be inputted in the model.
  """
  return cv2.resize(
    image,
    (IMAGE_WIDTH, IMAGE_HEIGHT),
    cv2.INTER_AREA
  )

def convert_rgb_to_yuv(image):
  return cv2.cvtColor(
    image, cv2.COLOR_RGB2YUV
  )

def preprocess_image(image):
  """ Combination of crop, resize and RGB to YUV conversion operations.
  """
  return convert_rgb_to_yuv(
    resize_image(
      crop_image(image)
    )
  )

def choose_image(data_dir, center, left, right, steering_angle):
  """ Choose random image from center, left or right and adjust steering angle.
  """
  choice = np.random.choice(3)
  if choice == 0:
    return load_image(data_dir, left), steering_angle + 0.2
  elif choice == 1:
    return load_image(data_dir, right), steering_angle - 0.2
  return load_image(data_dir, center), steering_angle
