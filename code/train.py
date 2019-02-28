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
parser.add_argument('-grad', '--grad_iterations', help="number of iterations of gradient descent", default=25, type=int)
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
parser.add_argument('-path','--path',default="../Digital/digital_Dave_curve1",type=str)
parser.add_argument('-direction','--direction',default='left',type=str)
parser.add_argument('-batch','--batch',default=5,type=int)
parser.add_argument('-type','--type',default = "jpg",choices=["jpg","png"])
parser.add_argument('-op','--op',default = False,type=bool)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

#28,16

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
#model2 = Dave_norminit(input_tensor=input_tensor, load_weights=True)
#model3 = Dave_dropout(input_tensor=input_tensor, load_weights=True)
# init coverage table
#model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)
model_layer_dict1 = init_coverage_tables2(model1)

# ==============================================================================================
# start gen inputs
#img_paths = image.list_pictures('./testing/center', ext='jpg')
#((0,0) is on the left-top side, = ( y, x) = (height,width))

#IO process for the final physical test
start_points = []
occl_sizes = []
imgs = []
angle3 = []
count=0
filelist = glob.glob(os.path.join(args.path,'*.'+args.type))
print("--------IMG READ-------")
for f in sorted(filelist):
    #TODO:!COUNT USED
    count+=1
    if(count==329 or count%5!=0):
        continue
    orig_name = f
    img = image.load_img(orig_name)
    input_img_data = image.img_to_array(img)
    orig_shape = np.shape(input_img_data)
    input_img_data = cv2.resize(input_img_data, dsize=(100, 100))
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    #TODO!
    input_img_data[:,:,:,0] += 0
    input_img_data[:,:,:,1] += 0
    input_img_data[:,:,:,2] += 0
    imgs.append(input_img_data)
print("--------IMG READ COMPLETE-------")
print("totally,", len(imgs)," imgs are read")
print("--------COORDINATES READ-------")
count=0
img_size = [orig_shape[0],orig_shape[1]]
with open(args.path+'/coordinates.txt') as f:
    for line in f:
        count+=1
        if(count%5!=0):
            continue
        row = [float(x) for x in line.split()]
        start_points.append([round(float(row[2])*100.0/img_size[0]),round(float(row[1])*100.0/img_size[1])])
        #TODO:TIMES 2!
        occl_sizes.append([max(1,round((float(row[8])-float(row[2]))*100.0/img_size[0]))*2,max(1,round((float(row[7])-float(row[1]))*100.0/img_size[1]))*2])
        count+=1
print("--------COORDINATES READ COMPLETE-------")
print("--------MODEL CONSTRUCTION -------")
for img in imgs:
    angle3.append(model1.predict(img)[0])
layer_name1, index1 = neuron_to_cover(model_layer_dict1)

# construct joint loss function
if args.target_model == 0:
    #loss1 = -args.weight_diff * K.mean(model1.get_layer('before_prediction').output[..., 0])
    loss1 = args.weight_diff * K.mean(model1.get_layer('prediction').output)
elif args.target_model == 1:
    loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
elif args.target_model == 2:
    loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1]) 
layer_output = loss1

# for adversarial image generation
final_loss = K.mean(layer_output)
if(args.direction=="left"):
    final_loss = K.mean(layer_output)
elif(args.direction=="right"):
    final_loss = -K.mean(layer_output)
else:
    print("LOSS EROOR!")
    exit()
# we compute the gradient of the input picture wrt this loss
grads = normalize(K.gradients(final_loss, input_tensor)[0])
#grads = normalize(K.gradients(loss1, input_tensor)[0])

# this function returns the loss and grads given the input picture
iterate = K.function([input_tensor], [loss1, loss1_neuron, grads])
print("--------MODEL CONSTRUCTION COMPLETE-------")
print("--------TRAINNING START-------")
logo_width = 600
logo_height = 400
logo = np.zeros((logo_height,logo_width,3))
if(args.op):   
    logo[:,:] = gen_optimal(imgs,model1,angle3,start_points,occl_sizes)

batch = 5
indexs = np.arange(len(imgs),dtype=np.int32)
imgs = np.array(imgs)
tmp_imgs = imgs.copy()
#last_diff : the total difference in last minibatch
last_diff = 0
#change_times : the total change times of logo in ONE ITERATION
change_times = 0
bad_change_times = 0
# we run gradient ascent for 20 steps
fixed_pixels = np.zeros_like(logo)
for iters in range(args.grad_iterations):
    fixed_pixels = np.zeros_like(logo)
    change_times = 0
    bad_change_times = 0
    if(args.greedy_stratage !='sequence_fix'):
        np.random.shuffle(indexs)
    for i in range(0,len(imgs),batch):
        if((args.greedy_stratage=='sequence_fix' or 'random_fix' or 'highest_fix') and i>args.fix_p*len(imgs)):
            break
        if i <= len(imgs) - batch:
            minibatch = [imgs[indexs[j]] for j in range(i,i+batch)]
        else:
            minibatch = [imgs[indexs[j]] for j in range(i,len(imgs))]
        logo_data = np.zeros((batch,logo_height,logo_width,3))
        count = 0
        for gen_img in minibatch:
            loss_value1, loss_neuron1, grads_value = iterate ([gen_img])
            if args.transformation == 'light':
                grads_value = constraint_light(grads_value)  # constraint the gradients value
            elif args.transformation == 'occl':
                #print(np.shape(grads_value),start_points[indexs[i+count]],occl_sizes[indexs[i+count]])
                grads_value = constraint_occl(grads_value, start_points[indexs[i+count]],
                                                occl_sizes[indexs[i+count]])  # constraint the gradients value
            elif args.transformation == 'blackout':
                grads_value = constraint_black(grads_value)  # constraint the  gradients value
            if(args.jsma):
                k_th_value = find_kth_max(grads_value,args.jsma_n)
                super_threshold_indices = abs(grads_value) < k_th_value
                grads_value[super_threshold_indices] = 0
            #IF the selected image's change make a positive reflection (diff in one image > 0.1) then
            #  we will count the image(add the image's gradient into the logo_data)
            # if angle_diverged3(angle3[indexs[i+count]],model1.predict(tmp_img)[0]):
            logo_data = transform_occl3(grads_value,start_points[indexs[i+count]],occl_sizes[indexs[i+count]],logo_data,count)
            #print(i,count,np.array_equal(np.sum(logo_data,axis = 0),np.zeros_like(np.sum(logo_data,axis = 0))))
            if(args.greedy_stratage=='random_fix' or args.greedy_stratage=='sequence_fix'): #random_fix and sequence fix is almost same except that the indexes are shuffled or not
                logo_data[count] = cv2.multiply(logo_data[count],1-fixed_pixels) #grads_value will only be adopted if the pixel is not fixed
                grads_value = np.array(logo_data[count],dtype=np.bool)
                grads_value = np.array(grads_value,dtype=np.int)
                fixed_pixels += grads_value
            count+= 1
        if(args.overlap_stratage=='sum'):
            logo_data = np.sum(logo_data,axis = 0)
        if(args.overlap_stratage=='max'):
            index = np.argmax(np.absolute(logo_data),axis=0)
            shp = np.array(logo_data.shape)
            dim_idx = []
            dim_idx.append(index)
            dim_idx += list(np.ix_(*[np.arange(i) for i in shp[1:]]))
            logo_data= logo_data[dim_idx]
        #TODO1: ADAM May be adapted.
        #TODO2: Smooth box constait    
        #TODO3: Consider the angle increase or decrease direction (the gradient should be positive or negative)

        tmp_logo = logo_data * args.step + logo
        tmp_logo = control_bound(tmp_logo)
        tmp_imgs = update_image(tmp_imgs,tmp_logo,start_points,occl_sizes) 
        # If this minibatch generates a higher total difference we will consider this one.
        this_diff = total_diff(tmp_imgs,model1,angle3)
        #print("iteration ",iters,". batch count ",i,". this time diff ",this_diff,". last time diff ", last_diff)
        if( this_diff> last_diff):
            logo += logo_data * args.step
            logo = control_bound(logo)
            imgs = update_image(imgs,logo,start_points,occl_sizes)
            last_diff = this_diff 
            change_times += 1
        else:
        #simulated_annealing is applied in current version. DATE: 26/07
            if(args.simulated_annealing):
                #if(this_diff != last_diff):
                    #print(i,"probability = ",pow(math.e,args.sa_k * (this_diff-last_diff)/(pow(args.sa_b,iters))),". this diff ",this_diff,". last diff ", last_diff)
                if(random.random() < pow(math.e,args.sa_k * (this_diff-last_diff)/(pow(args.sa_b,iters))) and this_diff != last_diff):
                    logo += logo_data * args.step
                    logo = control_bound(logo)
                    imgs = update_image(imgs,logo,start_points,occl_sizes)
                    last_diff = this_diff 
                    bad_change_times += 1
    angle_diff = 0  
    gray_angle_diff = 0   
    for i in range(len(imgs)): 
        angle1 = model1.predict(imgs[i])[0]
        gray_angle_diff += abs(angle1 - angle3[i])
        #if(i==30):
            #gen_img_deprocessed = draw_arrow3(deprocess_image(imgs[i]),angle3[i],angle1)
            #imsave('./generated_inputs/' +str(iters) + '_iter.png', gen_img_deprocessed)
    if(iters %5 == 0):
        print("iteration ",iters, ". diff between raw and adversarial", gray_angle_diff/len(imgs)*(180/math.pi),". change time is,",change_times,". bad_change_times,",bad_change_times)
    if(iters % 10 == 0):
        np.save('./train_output/'+str(iters)+'new_logo.npy',logo)
        imsave('./train_output/'+str(iters)+'new_logo.png', deprocess_image(logo,shape=(logo_height,logo_width,3)))
output = open('./train_output/'+"Output.txt","w")

print("--------TRAINNING COMPLETE-------")
print("--------GENERATE THE OUTPUT-------")
for i in range(len(imgs)):
    angle1 = model1.predict(imgs[i])[0]
    output.write(str(i)+" " + str(float(angle3[i]))+" "+str(float(angle1)) + "\n")
    gen_img_deprocessed = draw_arrow3(deprocess_image(imgs[i]),min(max(angle3[i],-math.pi/2),math.pi/2),angle1)
    imsave('./train_output/'+str(i) + 'th_img.png', gen_img_deprocessed)   
output.close()
