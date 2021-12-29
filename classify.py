# Python program to read
# image using PIL module
  
# importing PIL
from PIL import Image
from torchvision import models
from torchvision import transforms
#from mobilenetv2 import *

import sys
import torch 
import os

def main():
    try: 
        path_of_the_directory = sys.argv[1] 
        ext = ('.png', '.jpg')
        for file in os.listdir(path_of_the_directory):
            #print(file)
            if not file.endswith(ext):
                continue
            
            #print('Image file: start')
            if len(sys.argv) != 4: 
                raise ValueError("Input Arguments are Invalid")

            if sys.argv[2] == 'a':  
                net = models.alexnet(pretrained=True)        
            elif sys.argv[2] == 'r': 
                net = models.resnet101(pretrained=True)
            #elif sys.argv[2] == 'm':
                #net = mobilenetv2()
                #net.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))
                #net = torch.hub.load('pytorchvision:v0.10.0', 'mobilenet_v2', pretrained=True)
                #models.MobileNetV2(pretrained=True)
            else:
                raise ValueError("Choice Invalid") 

            transform = transforms.Compose([            #[1]
                transforms.Resize(256),                    #[2]
                transforms.CenterCrop(224),                #[3]
                transforms.ToTensor(),                     #[4]
                transforms.Normalize(                      #[5]
                mean=[0.485, 0.456, 0.406],                #[6]
                std=[0.229, 0.224, 0.225]                  #[7]
                )])

            #Relative Path
            img = Image.open(os.path.join(path_of_the_directory, file)).convert('RGB')

            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            net.eval()

            out = net(batch_t)
            #print(out.shape)

            with open('classlabels.txt') as f:
                classes = [line.strip() for line in f.readlines()]

            _, index = torch.max(out, 1)

            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

            # we need to print if the argument is in the classes
            if sys.argv[3] in classes[index[0]]:
                print(classes[index[0]] + "," + file) #, percentage[index[0]].item())    
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    main()