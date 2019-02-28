import cv2
import math
import random
from collections import defaultdict
import csv
import numpy as np
import tensorflow as tf
import skimage.transform as sktransform
import matplotlib.image as mpimg
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
#test  
def get_start_point(orig_pixels,orig_shape,final_shape):
    final_pixels = [round(float(orig_pixels[0])*final_shape[0]/orig_shape[0]),round(float(orig_pixels[1])*final_shape[1]/orig_shape[1])]
    return final_pixels

def get_occl_size(orig_start,orig_end,orig_shape,final_shape):
    occl_size = [max(1,round((float(orig_end[0])-float(orig_start[0]))*final_shape[0]/orig_shape[0])),max(1,round((float(orig_end[1])-float(orig_start[1]))*final_shape[1]/orig_shape[1]))]
    return occl_size
def v3_preprocess(image, top_offset=.375, bottom_offset=.125,shape=(32,128)):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    if(bottom == 0):
        bottom = 1
    image = sktransform.resize(image[top:-bottom, :], shape)
    return image

def draw_arrow2(img, angle1,thickness = 1):
    pt1 = (int(img.shape[1] / 2), img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (255, 0, 0), thickness)
    return img

def draw_arrow3(img, angle1,angle2):
    pt1 = (int(img.shape[1] / 2), img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 5)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 5)
    return img
def raw_draw_arrow3(img, angle1,angle2):
    pt1 = (int(img.shape[1] / 2), img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (1, 0, 0), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 1, 0), 1)
    return img


def draw_arrow(img, angle1, angle2, angle3):
    pt1 = (int(img.shape[1] / 2), img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    pt2_angle3 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle3)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle3)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle3, (255, 0, 0), 1)
    return img


def angle_diverged(angle1, angle2, angle3):
    if (abs(angle1 - angle2) > 0.2 or abs(angle1 - angle3) > 0.2 or abs(angle2 - angle3) > 0.2) and not (
                (angle1 > 0 and angle2 > 0 and angle3 > 0) or (
                                angle1 < 0 and angle2 < 0 and angle3 < 0)):
        return True
    return False

def angle_diverged2(angle1, angle2):
    if (abs(angle1 - angle2) > 0.1) and not (
                (angle1 > 0 and angle2 > 0 ) or (angle1 < 0 and angle2 < 0 )):
        return True
    return False

def angle_diverged3(angle1, angle2):
    if (abs(angle1 - angle2) > 0.1):
        return True
    return False


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data

def preprocess_image_v3(img_path,top_offset,bottom_offset):
    orig_name = image.load_img(img_path)
    orig_name = image.img_to_array(orig_name)
    orig_name = orig_name.astype(np.uint8)
    gen_img = v3_preprocess(orig_name,top_offset = top_offset,bottom_offset = bottom_offset)
    gen_img = np.expand_dims(gen_img, axis=0)
    return gen_img

def preprocess_image_v2(img_path,top_offset,bottom_offset):
    orig_name = image.load_img(img_path)
    orig_name = image.img_to_array(orig_name)
    orig_name = orig_name.astype(np.uint8)
    gen_img = v3_preprocess(orig_name,top_offset = top_offset,bottom_offset = bottom_offset,shape=(66,200))
    gen_img = np.expand_dims(gen_img, axis=0)
    return gen_img
def deprocess_image(x,shape=(100,100,3)):
    tmp = x.copy()
    tmp = tmp.reshape(shape)
    # Remove zero-center by mean pixel
    tmp[:, :, 0] += 103.939
    tmp[:, :, 1] += 116.779
    tmp[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    tmp = tmp[:, :, ::-1]
    tmp = np.clip(tmp, 0, 255).astype('uint8')
    return tmp


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    return K.truncated_normal(shape, stddev=0.1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

def init_coverage_tables2(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1



def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True

def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False

def transform_occl(gradients,start_point,rect_shape,logo_data,order):
    new_grads = np.zeros((np.shape(gradients)[0],rect_shape[0],rect_shape[1],np.shape(gradients)[3]))
    new_grads = gradients[:, start_point[0]:start_point[0] + rect_shape[0],start_point[1]:start_point[1] + rect_shape[1]]
    for i in range(rect_shape[0]):
        for j in range(rect_shape[1]):          
            logo_shape = np.shape(logo_data)
            logo_data[order,round(logo_shape[1]*1.00*i/(rect_shape[0])),round(logo_shape[2]*1.00*j/(rect_shape[1])),:] = new_grads[0,i,j,:]
    return logo_data

#for pixel transform:  nearest interpolation
def transform_occl2(gradients,start_point,rect_shape,logo_data,order):
    new_grads = np.zeros((np.shape(gradients)[0],rect_shape[0],rect_shape[1],np.shape(gradients)[3]))
    new_grads = gradients[:, start_point[0]:start_point[0] + rect_shape[0],start_point[1]:start_point[1] + rect_shape[1]]
    logo_shape = np.shape(logo_data)
    for i in range(logo_shape[1]):
        for j in range(logo_shape[2]):          
            logo_data[order,i,j,:] = new_grads[0,round(rect_shape[0]*1.00*i/(logo_shape[1])-0.5),round(rect_shape[1]*1.00*j/(logo_shape[2])-0.5),:]
    return logo_data

#for pixel transform:  linear interpolation
def transform_occl3(gradients,start_point,rect_shape,logo_data,order):
    new_grads = np.zeros((np.shape(gradients)[0],rect_shape[0],rect_shape[1],np.shape(gradients)[3]))
    new_grads = gradients[:, start_point[0]:start_point[0] + rect_shape[0],start_point[1]:start_point[1] + rect_shape[1],:]
    logo_shape = np.shape(logo_data)
    #In this version (29/07/2018), we do not use own rescale code but opencv resize instead
    #print(np.shape(new_grads))
    logo_data[order] = cv2.resize(new_grads[0], dsize=(logo_shape[2], logo_shape[1]))
    '''
    for i in range(logo_shape[1]):
        for j in range(logo_shape[2]):
            y_approximation = rect_shape[0]*1.00*i/(logo_shape[1])
            x_approximation = rect_shape[1]*1.00*j/(logo_shape[2])
            y_offset =  y_approximation - int(y_approximation)
            x_offset = x_approximation -int(x_approximation)
            logo_data[order,i,j,:] = new_grads[0,int(y_approximation),int(x_approximation),:] * (1-x_offset)*(1-y_offset) + \
                                     new_grads[0,min(rect_shape[0]-1,int(y_approximation)+1),int(x_approximation),:] * (1-x_offset)*(y_offset) + \
                                     new_grads[0,int(y_approximation),min(rect_shape[1]-1,int(x_approximation)+1),:] * (x_offset)*(1-y_offset) + \
                                     new_grads[0,min(rect_shape[0]-1,int(y_approximation)+1),min(rect_shape[1]-1,int(x_approximation)+1),:] * (x_offset)*(y_offset)
    '''
    return logo_data

def transfrom_accurate(gradients,des_pixel,logo_data,order):
    des = np.array([[0.1, 0], [np.shape(logo_data[0])[1]-1, 0], [0, np.shape(logo_data[0])[0]-1], [np.shape(logo_data[0])[1]-1, np.shape(logo_data[0])[0]-1]],np.float32)
    transform = cv2.getPerspectiveTransform(des_pixel,des)
    logo_data[order] = cv2.warpPerspective(gradients,transform,( np.shape(logo_data[0])[1], np.shape(logo_data[0])[0]))
    return logo_data
def update_image(imgs,logo,start_point,occl_size):
    for i in range(len(imgs)):
        imgs[i][0,start_point[i][0]:start_point[i][0]+occl_size[i][0],start_point[i][1]:start_point[i][1]+occl_size[i][1],:] = cv2.resize(logo,(occl_size[i][1],occl_size[i][0]))[:min(occl_size[i][0],np.shape(imgs[i])[1]-start_point[i][0]),:min(occl_size[i][1],np.shape(imgs[i])[2]-start_point[i][1])]
    return imgs
#def accurate_logo_grad(gradients,des_pixels):

def accurate_update(imgs,logo,des_pixels):
    src = np.array([[0, 0], [np.shape(logo)[1]-1, 0], [0, np.shape(logo)[0]-1], [np.shape(logo)[1]-1, np.shape(logo)[0]-1]],np.float32)
    for i in range(len(imgs)):
        transform = cv2.getPerspectiveTransform(src,des_pixels[i])
        output = cv2.warpPerspective(logo,transform,( np.shape(imgs[i])[2], np.shape(imgs[i])[1]),flags=cv2.INTER_LINEAR )
        mask = output.astype(np.bool)
        mask = mask.astype(np.float32)
        back = cv2.multiply(1.0-mask,imgs[i][0])
        imgs[i][0] = np.array(cv2.add(back,np.array(output,dtype=np.float32)))
    return imgs

def control_bound(logo):
    np.clip(logo[:,:,0], -103.939, 255-103.939, out=logo[:,:,0])
    np.clip(logo[:,:,1], -116.779, 255-116.779, out=logo[:,:,1])
    np.clip(logo[:,:,2], -123.68, 255-123.68, out=logo[:,:,2])
    return logo

def raw_control_bound(logo):
    np.clip(logo, 0, 1, out=logo)
    return logo
def preprocess_color(color):
    color[0] = color[0] -103.939
    color[1] = color[1] -116.779
    color[2] = color[2] - 123.68
    return color

def total_diff(imgs,model1,angles_2):
    angles_diff = []
    count = 0
    for img in imgs:
        angles_diff.append(abs(model1.predict(img)[0]-angles_2[count]))
        count+=1
    return sum(angles_diff)

def gen_optimal(imgs,model1,angle3,start_points,occl_sizes):
    logo = np.zeros((480,640,3))
    result = {"pixel_value":np.zeros([10,3]),"diff":np.zeros(10)}
    for blue in range(0,256,51):
        for green in range(0,256,51):
            #print("[",blue,green,":] claculated, current result is,",result)
            for red in range(0,256,51):
                logo[:,:] = preprocess_color([blue,green,red])
                #imgs = accurate_update(imgs,logo,des_pixels)
                imgs = update_image(imgs,logo,start_points,occl_sizes)
                this_diff = total_diff(imgs,model1,angle3)
                this_diff = this_diff/len(imgs) * 180 / math.pi
                if(this_diff > result["diff"][0]):
                    result["pixel_value"][0] = np.array([blue,green,red])
                    result["diff"][0] =  this_diff
                    index = np.argsort(result["diff"])
                    result["diff"] = result["diff"][index]
                    result["pixel_value"] = result["pixel_value"][index]
    print(result)
    return preprocess_color(result["pixel_value"][-1])

def raw_gen_optimal(imgs,model1,angle3,start_points,occl_sizes):
    logo = np.zeros((480,640,3))
    result = {"pixel_value":np.zeros([10,3]),"diff":np.zeros(10)}
    for blue in range(0,256,51):
        for green in range(0,256,51):
            #print("[",blue,green,":] claculated, current result is,",result)
            for red in range(0,256,51):
                logo[:,:] = [blue/255.0,green/255.0,red/255.0]
                #imgs = accurate_update(imgs,logo,des_pixels)
                imgs = update_image(imgs,logo,start_points,occl_sizes)
                this_diff = total_diff(imgs,model1,angle3)
                this_diff = this_diff/len(imgs) * 180 / math.pi
                if(this_diff > result["diff"][0]):
                    result["pixel_value"][0] = np.array([blue/255.0,green/255.0,red/255.0])
                    result["diff"][0] =  this_diff
                    index = np.argsort(result["diff"])
                    result["diff"] = result["diff"][index]
                    result["pixel_value"] = result["pixel_value"][index]
    print(result)
    return result["pixel_value"][-1]

def read_input(imgs,des_pixels,occl_sizes,start_points,filelist,coordination,is_crop = False,is_all = False):
    if(is_crop):
        img_size = [600,800]
    else:
        img_size = [1080,1920]
    for f in sorted(filelist):
        orig_name = f
        gen_img = preprocess_image(orig_name)
        imgs.append(gen_img)
    with open(coordination) as f:
        if(is_all):
            spamreader = csv.reader(f)
        else:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spamreader:
            if(len(row) != 9):
                print(len(row))
                continue
            tmp = np.array([[float(row[2])*100.0/img_size[1],float(row[1])*100.0/img_size[0]],[float(row[6])*100.0/img_size[1],float(row[5])*100.0/img_size[0]],[float(row[4])*100.0/img_size[1],float(row[3])*100.0/img_size[0]],[float(row[8])*100.0/img_size[1],float(row[7])*100.0/img_size[0]]],np.float32)
            des_pixels.append(tmp)
            start_points.append([round(float(row[1])*100.0/img_size[0]),round(float(row[2])*100.0/img_size[1])])
            occl_sizes.append([round((float(row[7])-float(row[1]))*100.0/img_size[0]),round((float(row[8])-float(row[2]))*100.0/img_size[1])])
    print(coordination + " read complete")
    return imgs,des_pixels,occl_sizes,start_points
def find_kth_max(array,k):
    tmp = array.flatten()
    tmp = abs(tmp)
    tmp.sort()
    return tmp[-k]
