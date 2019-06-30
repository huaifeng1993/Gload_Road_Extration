from __future__ import print_function ,division
import os
import torch
from skimage import io,transform
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import warnings
import cv2
import random
warnings.filterwarnings("ignore")

MEAN=[0.410,0.383,0.288]
STD=[0.156,0.126,0.123]
class RoadExtrationDataset(Dataset):
    """ INTRODUCTION:
        https://competitions.codalab.org/competitions/18467

        @InProceedings{DeepGlobe18,
        author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum,
        David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, 
        Forest and Tuia, Devis and Raskar, Ramesh},title = {DeepGlobe 2018: 
        A Challenge to Parse the Earth Through Satellite Images},booktitle 
        = {The IEEE Conference on Computer Vision and Pattern Recognition 
        (CVPR) Workshops},month = {June},year = {2018}
        }

        DATA:
            The training data for Road Challenge contains 6226 statellite imagery 
        in RGB,size 1024X1024.
        Label:
            * Each satellite image is paired with a mask image for road labels. The 
            mask is a grayscale image, with white standing for road pixel, and 
            black standing for background.

            * File names for satellite images and the 
            corresponding mask image are <id>_sat.jpg and <id>_mask.png. <id> is a 
            randomized integer.

            * Please note:
                ** The values of the mask image may not be pure 0 and 255. When 
                converting to labels, please binarize them at threshold 128.
                ** The labels are not perfect due to the cost for annotating 
                segmentation mask, specially in rural regions. In addition, we 
                intentionally didn't annotate small roads within farmlands.
    """
    def __init__(self,root_dir,n_train,random_seed=32,val=False,transform=None):
        """
            Args:
        """
        self.class_info=[]
        self.img_h=1024
        self.img_w=1024
        self.new_img_h=512
        self.new_img_w=512
        self.transform=transform
        self.val=val
        self.n_train = n_train

        images=os.listdir(root_dir)
        images.sort()
        for x in range(0,len(images),2):
            id=images[x][:-9]
            img_source=id+"_sat.jpg"
            label_source=id+"_mask.png"
            self.class_info.append({"img_source":os.path.join(root_dir,img_source),
                                "label_source":os.path.join(root_dir,label_source)})

        random.seed(random_seed)
        random.shuffle(self.class_info)

        if self.val:
            self.class_info = self.class_info[self.n_train:]
        else:
            self.class_info = self.class_info[:self.n_train]

    def __len__(self):
        """

        """
        return len(self.class_info)
    
    def __getitem__(self,idx):

        img=io.imread(self.class_info[idx]["img_source"])
        label=io.imread(self.class_info[idx]["label_source"])[:,:,0]
        label[np.where(label>128)]=1
        label[np.where(label<0)]=0

        sample={"image":img,"label":label} 
        if self.transform:
            sample=self.transform(sample)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        # resize img and label without interpolation (want the image to still match
        # label_img, which we resize below):
        image=cv2.resize(image,(self.output_size,self.output_size),
                        interpolation=cv2.INTER_NEAREST)
        label=cv2.resize(label,(self.output_size,self.output_size),
                        interpolation=cv2.INTER_NEAREST)
        return {'image':image,'label':label}
        
class RandomFlip(object):
    """
        flip the image and the label with 0.5 probability
    """
    def __call__(self,sample):
        flip = np.random.randint(low=-1,high=2)
        image,label=sample['image'],sample['label']
        if flip ==1:
            image=cv2.flip(image,1)
            label=cv2.flip(label,1)
        if flip ==0:
            image=cv2.flip(image,0)
            label=cv2.flip(label,0)
        if flip ==-1:
            image=cv2.flip(image,-1)
            label=cv2.flip(label,-1)
        return {'image':image,'label':label}

class RandomScale(object):
    """Random rescale the image in a sample to a given scale_scope.
    Args:
        randomly scale the img and the label
    """
    def __init__(self, scale_scope):
        assert isinstance(scale_scope, tuple)
        self.scale_scope = scale_scope
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        image_h=image.shape[1]
        image_w=image.shape[0]
        ###################################################
        # randomly scale the img and the label:
        ###################################################
        scale = np.random.uniform(low=self.scale_scope[0],high=self.scale_scope[1])
        new_img_h=int(scale*image_h)
        new_img_w=int(scale*image_w)
        # resize img and label without interpolation (want the image to still match
        # label_img, which we resize below):
        image=cv2.resize(image,(new_img_w,new_img_h),
                        interpolation=cv2.INTER_NEAREST)
        label=cv2.resize(label,(new_img_w,new_img_h),
                        interpolation=cv2.INTER_NEAREST)
        return {'image':image,'label':label}

class  RandomCorp(object):
    """
        select a NXN random crop from the img and label
    """
    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        image_h=image.shape[1]
        image_w=image.shape[0]
        start_x=np.random.randint(low=0,high=(image_w-self.crop_size))
        end_x=start_x+self.crop_size
        start_y=np.random.randint(low=0,high=(image_h-self.crop_size))
        end_y=start_y+self.crop_size
        image = image[start_y:end_y,start_x:end_x]
        label=label[start_y:end_y,start_x:end_x]
        return {'image':image,'label':label}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
        Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
        will normalize each channel of the input ``torch.*Tensor`` i.e.
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        image=image/255
        image=image-np.array(self.mean)#subtract mean value of the dataset then divided the std
        image=image/np.array(self.std)
        image=image.astype(np.float32)
        return {'image':image,'label':label}

class RandomRotate(object):

    def __call__(self, sample):
        
        rand=np.random.randint(low=-1,high=3)
        if rand == -1:
            sat_img = np.rot90(sample['image'], k=1)
            map_img = np.rot90(sample['label'], k=1)

        elif rand == 0:
            sat_img = np.rot90(sample['image'], k=2)
            map_img = np.rot90(sample['label'], k=2)

        elif rand == 1:
            sat_img = np.rot90(sample['image'], k=3)
            map_img = np.rot90(sample['label'], k=3)

        elif  rand == 2:
            sat_img = sample['image']
            map_img = sample['label']
            
        return {'image': sat_img.copy(), 'label': map_img.copy()}

class ToTensor(object):
    """
        Conver ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        label = torch.unsqueeze(label,dim=0)
        return {'image': image,'label': label}


if __name__=="__main__":
    dataset=RoadExtrationDataset(root_dir="./data/RoadExtraction/train",
                                 n_train = 5000,
                                 random_seed=32,
                                 transform=transforms.Compose([Rescale(512),
                                                               RandomFlip(),
                                                               RandomScale((0.75,1.2)),
                                                               RandomCorp(256),
                                                               RandomRotate(),
                                                               #Normalize([0.410,0.383,0.288],[0.156,0.126,0.123]),
                                                               #ToTensor(),
                                                               ],))
    fig=plt.figure()
    print(len(dataset))
    for i in range(len(dataset)):
        sample=dataset[i]
        print(i,sample["image"].shape,sample['label'].shape)
        ax=plt.subplot(1,2,1)
        plt.tight_layout()
        ax.axis('off')
        plt.imshow(sample["label"])

        ax=plt.subplot(1,2,2)
        plt.tight_layout()
        ax.axis('off')
        plt.imshow(sample["image"])
        plt.show()
        if i==1:
            break