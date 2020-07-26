import os
import glob
import argparse
import time
from PIL import Image
import numpy as np
import PIL
import gc

import multiprocessing as mp

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from matplotlib import pyplot as plt

import skimage
from skimage.transform import resize

from tqdm import tqdm

# Argument Parser
#parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
#parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
#parser.add_argument('--input', default='my_examples', type=str, help='Input filename or folder.')
#parser.add_argument('--image_output', default='my_out_examples', type=str, help='Output filename or folder.')
#args = parser.parse_args()

iteration = 6

args = {"model": "nyu.h5",
        "input": "/content/content/fg_bg_dataset/bgfg_overlay/",
        "output": "/content/content/fg_bg_dataset/bgfg_depth" + str(iteration) + "/"}

# Custom object needed for inference and training
start = time.time()



# Load model into GPU / CPU
def load_images_with_resize(image_files):
    loaded_images = []
    for file in image_files:
        with Image.open( file ) as im:
            im = im.resize((640, 480), PIL.Image.ANTIALIAS)
            x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
            loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def process_depth_estimate(model, input_folder, output_folder, file_pattern):
  custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
  print('Loading model...')
  model = load_model(model, custom_objects=custom_objects, compile=False)
  end = time.time()
  print('\nModel loaded ({0}) in time {1}.'.format(args["model"], end - start))

  pbar = tqdm(enumerate(glob.glob(input_folder + file_pattern)))
  for pidx, input_image in pbar:
    a = input_image.split('/')
    filename = a[len(a) - 1]

    pbar.set_description("Processing image: " + filename)
                         
    temp = filename.split('.')
    filename_w_ext = temp[0]
    # Input images
    with Image.open( input_image ) as im:
        im = im.resize((640, 480), PIL.Image.ANTIALIAS)
        x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
    inputs = np.stack([x], axis=0)

    # Compute results
    outputs = predict(model, inputs)
    # Display results
    plasma = plt.get_cmap('gray')
    output_copy = outputs.copy()
    for i in range(output_copy.shape[0]):
        imgs = []
        
        rescaled = output_copy[i][:,:,0]
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)
        matplotlib_image = plt.imshow(plasma(rescaled)[:,:,:3])

        pil_image = Image.fromarray(np.uint8( ( matplotlib_image.get_array()*255))).convert("L").resize((224,224))
        pil_image.save(os.path.join(output_folder, "depth_" + filename_w_ext + ".jpg"))
        plt.close()
        
    del inputs
    del outputs
    del output_copy
    del matplotlib_image

    if pidx == 1 or pidx % 400 == 0:
        gc.collect()
  del pbar

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

for k in range(iteration * 10, (iteration + 1) * 10):
    p = mp.Pool(processes =1)
    if k < 38:
        continue
    file_pattern1 = "ol_bg0" + str(k) + "*.png"
    #result = process_depth_estimate(args["model"], args["input"], args["output"], file_pattern1)
    print("Working on images: ", file_pattern1)
    result = p.apply_async(process_depth_estimate, args = (args["model"], args["input"], args["output"], file_pattern1), callback = log_result)
    p.close()
    p.join()
    end = time.time()
    print('Done. It took: ', end - start)
    print('\n')
