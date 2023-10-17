
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import timm

classes_ja = ["東武50090系", "メトロ10000系", "メトロ17000系", "東武8000系", "東武9000系","東武10000系","東武10030系","東武30000系","東武50000系"]
classes_en = ["tobu50090", "metro10000", "metro17000", "tobu8000", "tobu9000","tobu10000","tobu10030","tobu30000","tobu50000"]

img_size = 128

# 画像認識モデル
timm_model = "mobilenetv3_small_100"
net = timm.create_model(model_name=timm_model, pretrained=True, in_chans=3, num_classes=len(classes_ja))

# 訓練済みパラメータの読み込みと設定
net.load_state_dict(torch.load("model_cnn_"+timm_model+".pth", map_location=torch.device("cpu")))

def predict(img):
    transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                                   ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size)
    
    # 予測
    net.eval()
    y = net(x)
    
    # 結果を返す
    y_pred = F.softmax(torch.squeeze(y))
    sorted_prob, sorted_indices = torch.sort(y_pred, descending=True) # 予測確率を降順にソート
    
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
        
