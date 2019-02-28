# Deep Billboard

This repository is for providing more details for replicating our experiments and helping you understand our approach better.

Also, we provide some scripts for you to demonstrate the training and evaluation and reproduce some experimental results.

## Repository structure

+ Digital

  This directory stores the data required by our approach to train a digital billboard, we provide some cases in several common driving datasets. In each subdirectory, a series of images 

+ Physical

  this subdirectory stores the data obtained by our physical cameras. 

  + white

    this folder includes a series of images, where the  billboard is white for .

  + green

    this folder includes a series of images, where the  billboard is obtained/trained by our approach.

+ code

  this subdirectory includes some scripts for demonstrating our experiments

## Demon

You should be noticed that **python3** is used as our programming language in this project and some packages are needed.

Firstly, go to the scripts folder:

```
cd code
```

### Train the model to generate a billboard

#### Usage

You can train a billboard and view the digital adversarial results easily by typing following command:

```
python3 train.py
```

Also, there are several arguments you can modify:

```
python3 train.py -direction='right' -path='../Digital/digital_Dave_straight1'
```

`-direction` can be `'right'`' or `'left'`, which specifies the behavior of the adversarial direction result we want, `'left'` is in default.

`-path` is the digital case path, it can be modified to test different cases.

#### Results explanation

The results of trainning process are located in `./train_output`, each element should be explained as following:

+ `%step new_logo.png/.npy` : the generated billboard of our algorithm, `step` means the trainning step.
+ `%order th_img.png` :  the digital results of our method ( replacing the original billboard with our adversarial billboard digitally), the green arrow is adversarial direction and the blue one is the direction with original billboard, `order` is the sequence number of image series.
+ `Output.txt` : the file records the direction of different billboards for future use.

### Test physical case

#### Usage

After generating a digital billboard, we need to print the billboard physically and see if it works properly. In this repository, we prepared the physical video using trained billboard in `../Physical/green/` folder for convenience and you can test the case by:

```
python3 test.py
```

Similarly, you can test another case for control experiments with white billboard by changing the `path` parameter:

```
python3 test.py -path='../Physical/white'
```

#### Result explanation

The results of physical case are located in `./test_output`, each element should be explained as following:

+ `%order.png` :  the physical results of our method, the red arrow is direction predicted by Dave model.

#### Video generation

You can also generate the video from the image series to visualize the result clearly by typing:

```
ffmpeg -i ./test_output/%03d.png green.avi
```

For your convenience, we have uploaded two videos corresponding to two billboards (white one and adversarial one)  and they are named as `white.avi` and `green.avi` respectively.