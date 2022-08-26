
'''
Copyright 2022 Andrea Rafanelli.
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on 
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Andrea Rafanelli'


import random
from skimage.measure import label, regionprops
import colorsys

from enum import Enum, unique
from PIL import Image

def _rgb(v):
    r, g, b = v[1:3], v[3:5], v[5:7]
    return int(r, 16), int(g, 16), int(b, 16)


@unique
class Mapbox(Enum):
    """Mapbox-themed colors.
    See: https://www.mapbox.com/base/styling/color/
    """

    dark = _rgb("#404040")
    gray = _rgb("#eeeeee")
    light = _rgb("#f8f8f8")
    white = _rgb("#ffffff")
    cyan = _rgb("#3bb2d0")
    blue = _rgb("#3887be")
    bluedark = _rgb("#223b53")
    denim = _rgb("#50667f")
    navy = _rgb("#28353d")
    navydark = _rgb("#222b30")
    purple = _rgb("#8a8acb")
    teal = _rgb("#41afa5")
    green = _rgb("#56b881")
    yellow = _rgb("#f1f075")
    mustard = _rgb("#fbb03b")
    orange = _rgb("#f9886c")
    red = _rgb("#e55e5e")
    pink = _rgb("#ed6498")


def make_palette(*colors):

    rgbs = [Mapbox[color].value for color in colors]
    flattened = sum(rgbs, ())
    return list(flattened)

    return palette
    

def reverse_transform_mask(inp):
    
    inp = inp.transpose((1, 2, 0))
    t_mask = np.argmax(inp,axis=2).astype('float32') 
    t_mask = cv2.resize(t_mask, dsize=(4000, 3000))
    
    return t_mask


class getData:
    
    def __init__(self, x_paths):
        
        self.x_paths = x_paths
        self.y_paths = [x.replace("image", "mask").replace(".jpg", "_lab.png") for x in self.x_paths]
        
    def __len__(self):
        
        return len(self.x_paths)
    

    def __getitem__(self, index):
        
        image = Image.open(self.x_paths[index])
        mask = Image.open(self.y_paths[index])
    
            
        return image, mask, index


def augmentation(image, mask):
    
    a1 = 90
    a2 = 180
    a3 = 270

    im1 = image.rotate(a1, expand = True)
    im2 = image.rotate(a2, expand = True)
    im3 = image.rotate(a3, expand = True)

    m1 = mask.rotate(a1, expand = True)
    m2 = mask.rotate(a2, expand = True)
    m3 = mask.rotate(a3, expand = True)
    
    return im1, im2, im3, m1, m2, m3
