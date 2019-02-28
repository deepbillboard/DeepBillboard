'''
usage: python gen_diff.py -h
'''

from __future__ import print_function
from os.path import basename
import argparse
import glob
import os
import random
import csv
import re
from scipy.misc import imsave

from driving_models import *
from utils import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument('-tr', '--transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'], default='occl')
parser.add_argument('-diff', '--weight_diff', help="weight hyperparm to control differential behavior", default=1, type=float)
parser.add_argument('-w', '--weight_nc', help="weight hyperparm to control neuron coverage", default=0.1, type=float)
parser.add_argument('-st', '--step', help="step size of gradient descent", default=5, type=float)
parser.add_argument('-sd', '--seeds', help="number of seeds of input", default=1, type=int)
parser.add_argument('-grad', '--grad_iterations', help="number of iterations of gradient descent", default=1000, type=int)
parser.add_argument('-th', '--threshold', help="threshold for determining neuron activated", default=0, type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(50, 50), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)
parser.add_argument('-overlap_stra','--overlap_stratage',help='max:select maximum gradient value. sum:..., highest: detect the influence', choices=['max','sum','highest_influence'],default = 'sum')
parser.add_argument('-greedy_stra','--greedy_stratage',help='random_fix:fix pixels if it is changed in one iteration and the img is selected randomly. dynamic:pixels can be changed in each iteration. highest: detect the influence', choices=['random_fix','sequence_fix','dynamic','hightest_fix'],default = 'dynamic')
parser.add_argument('-fix_p','--fix_p',help='parameter p(percetage) of fixed images(batches), only p of 1 images will be selected', default = 1.0,type=float)
parser.add_argument('-jsma','--jsma',help= 'Using jsma or not',default = False,type=bool)
parser.add_argument('-jsma_n','--jsma_n',help='parameter n of jsma, top n pixels in gradients of board will be selected', default = 5,type=int)
parser.add_argument('-sa','--simulated_annealing',help= 'Simulated annealing to control the logo update',default = True,type=bool)
parser.add_argument('-sa_k','--sa_k',help='parameter k of simulated annealing, higher = less changing probability', default = 30,type=int)
parser.add_argument('-sa_b','--sa_b',help='parameter b in range (0,1) of simulated annealing, higher = higher changing probability. p = pow(e,k*diff/pow(b,iter))', default = 0.96,type=float)
parser.add_argument('-gpu','--gpu',default=0,type=int)
parser.add_argument('-path','--path',default="../Physical/green",type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

#28,16

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model_layer_dict1 = init_coverage_tables2(model1)

# ==============================================================================================
# start gen inputs
#img_paths = image.list_pictures('./testing/center', ext='jpg')
#((0,0) is on the left-top side, = ( y, x) = (height,width))
acc = 0
filelist = glob.glob(os.path.join(args.path, '*.png'))#3.2828627


angle3 = []
imgs = []
raw_imgs = []
des_pixels = []
occl_sizes = []
start_points = []


ss= []
print("---IMG READ---")
for i in range(len(filelist)):
    s = re.findall("\d+",filelist[i])[0]
    ss.append(int(s))
ss = np.array(ss)
indexes = ss.argsort()
for i in sorted(ss):
    index = indexes[i]
    f = filelist[index]
    orig_name = f
    gen_img = preprocess_image(orig_name)
    raw_img = preprocess_image(orig_name,(1080,1920))
    imgs.append(gen_img)
    raw_imgs.append(raw_img)
imgs = np.array(imgs)

angle3 = [] 
print("---IMG READ COMPLETE---")
print("---CALCULATE THE DIRECTION---")
for i in range(len(imgs)):
    angle3.append(model1.predict(imgs[i])[0])

print("---CALCULATE THE DIRECTION COMPLETE---")
print("---IMG WRITE---")
for i in range(len(raw_imgs)): 
    #if(i<100):
    #   continue 
    #output.write(str(i)+" " + str(float(angle3[i]))+ "\n")
    #gen_img_deprocessed = draw_arrow2(deprocess_image(imgs[i]),angle3[i])
    gen_img_deprocessed = draw_arrow2(deprocess_image(raw_imgs[i],(1080,1920,3)),angle3[i],10)
    imsave('./test_output/'+str("%03d" % i) + '.png', gen_img_deprocessed) 
print("---IMG WRITE COMPLETE---")
