import tensorflow as tf

resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
)

ResNet = tf.keras.models.Model(inputs=resnet.input, outputs=resnet.layers[-1].output)

#ResNet.summary()
#import torch
#from pytorch_pretrained_vit import ViT
#VTransformer = ViT('B_16_imagenet1k', pretrained=True)
#VTransformer.fc = torch.nn.Identity()
#print(VTransformer)

