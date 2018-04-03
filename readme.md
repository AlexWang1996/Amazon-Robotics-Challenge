# Amazon Robotics Challenge     
## About the datasets
* The datasets come from UC Berkeley and are composed of 27 different things.
> sources can be found [here](http://rll.berkeley.edu/amazon_picking_challenge/)

## About the model
* The Convolutional Neural Network which I choose is the [VGG19](https://arxiv.org/abs/1409.1556) from  Oxford Visual Geometry Group
* Simply, I use the weights trained on [ImageNet](www.image-net.org/) to initialize the Network and retrain it for the ARC datasets
## Result
* Until now, I  trained Network under 27 classes.
* The Result given below is trained under 27 classes.
### Train Accuarcy
![train_accuracy](https://github.com/AlexWang1996/Amazon-Robotics-Challenge/train_acc.png)     
### Train Loss
![train_loss](https://github.com/AlexWang1996/Amazon-Robotics-Challenge/train_loss.png)          
### Valid Accuarcy
![valid_accuracy](https://github.com/AlexWang1996/Amazon-Robotics-Challenge/val_acc.png)  
### Valid Loss
![valid_loss](https://github.com/AlexWang1996/Amazon-Robotics-Challenge/val_loss.png)      
### Training Time on a single GTX 1080Ti
![training_time](https://github.com/AlexWang1996/Amazon-Robotics-Challenge/training_time.png)    
