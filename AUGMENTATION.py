import os
from shutil import copy2
import cv2
import numpy as np
from random import randint
path='dataset/emojinator_aug'
path2='dataset/emojinator_aug2'

for fol in os.listdir(path):
    os.mkdir(path2+'/'+fol)
    for image in os.listdir(path+'/'+fol):
        source=os.path.join(path,fol,image)
        des=os.path.join(path2,fol)
        copy2(source,des)
        cur_image=cv2.imread(os.path.join(des,image))
        new_filename,ext=os.path.splitext(image)
        npath=new_filename+'1'+ext #filename for flip
        final_path=des+'/'+npath
        flipim=np.fliplr(cur_image) #for flip
        cv2.imwrite(final_path,flipim)
        
        npath=new_filename+'2'+ext #filename for blur
        final_path=des+'/'+npath
        blurImg=cv2.blur(cur_image,(10,10))
        cv2.imwrite(final_path,blurImg)
        
        npath=new_filename+'4'+ext #filename for rotate
        final_path=des+'/'+npath
        r1=(randint(-25,25))
        h,w=cur_image.shape[:2]
        scale=1.0
        center=tuple(np.array([h,w])/2)
        M=cv2.getRotationMatrix2D(center,r1,scale)
        rotated=cv2.warpAffine(cur_image,M,(h,w))
        cv2.imwrite(final_path,rotated)
        
        npath=new_filename+'5'+ext #filename for exposure
        final_path=des+'/'+npath
        a=np.double(cur_image)
        r2=randint(-50,50)
        b=a+r2
        exposure=np.uint8(b)
        cv2.imwrite(final_path,exposure)
        
        npath=new_filename+'3'+ext #filename for noise
        final_path=des+'/'+npath
        row,col,ch=cur_image.shape
        s_vs_p=0.9
        amount=.04
        out=cur_image
        num_salt=np.ceil(amount*cur_image.size*s_vs_p)
        coords=[np.random.randint(0,i-1,int(num_salt)) for i in cur_image.shape]
        out[coords]=1
        num_pepper=np.ceil(amount*cur_image.size*(1-s_vs_p))
        coords=[np.random.randint(0,i-1,int(num_pepper)) for i in cur_image.shape]
        out[coords]=1
        cv2.imwrite(final_path,out)
                       