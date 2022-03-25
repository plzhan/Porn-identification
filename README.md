# SZ_DpL
This is an open source deep learning model. You can use the weights already provided by this model for free.
The model is allowed to be modified at will and any non-commercial use, this account does not assume any joint and several responsibility

## Porn-identification
The project provides trained neural network models for image classification (for pornography identification), including weights, model structure, etc...  But it does not includes the dataset.



****


We provide small models for **image identification**(5 classes: including Porn Identification)

The **first** category includes landscape and humanities

The **second** category contains cartoon category

The **third** category contains erotic lingerie

The **fourth** category contains cartoon pornography

The **fifth** category contains pornography


### install the package
```
! pip install -i https://test.pypi.org/simple/ SZ-DpL 
```

### Instruction

Total params: 21,868,677

Trainable params: 21,851,653

Non-trainable params: 17,024
```
import numpy as np
import tensorflow as tf
from SZ_DpL.PI_resnet34 import get_PI_model

# here, the weights have to be download on github yourself...
model = get_PI_model(image_size=224, class_num=5, load_weights=False)  #  set load_weights False, or you get ERROR

def Input_One_pics(image_path):
    """
    Convert single image to one array.
    :param image_path:
    :return:
    """
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    return input_arr
   
data = Input_One_pics(image_path)
model.predict(data)  # here input one img_array and its dim has to be [1, 224, 224, 3], 3 means three channels.  
```

