import numpy as np
from PIL import Image

def reshape_breakout(image, p=0.5):
  """ Provided with breakout image and percentage resize, 
  return square image with side lengths equal to the 
  samllest of the reshaped sides in uint8 format"""
  B = np.mean(image, axis=2)
  B = B / np.max(B) * 255
  B = B.astype(np.uint8)
  B = np.squeeze(B)
  B = B[31:195, 7:153]
  d1 = int(p*B.shape[0])
  d2 = int(p*B.shape[1])
  d = min(d1, d2)
  return np.array(Image.fromarray(B).resize((d,d), Image.NEAREST))
