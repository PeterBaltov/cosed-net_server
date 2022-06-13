import torch
import numpy as np

from segmentation_model import SegmentationModel
from classification_model import ClassificationModel

from PIL import Image

from single_image_prerocessing import pre_process_image

"""
The following code is used to predict if a given retinal image has the following lesions:
    - Hard Exudate(EX)
    - Hemorrhages(HE)
    - Microaneurysms(MA)
    - Soft Exudate(SE)
Addionally, the severity of the Diabetic Rethinopathy is predicted.
This is done via 4 neural networks for the 4 aforementioned lesion segmentation tasks 
and 1 neural network for the severity prediction.
"""

# Selecting the paths to the weights of the different models
CKPT_CLASSIFICATION = r"/home/nvidia/Desktop/3_WebDev/server/classification_weights/fgadr_csv_v2.ckpt"
CKPT_EX = r"/home/nvidia/Desktop/3_WebDev/server/segmentation_weights/EX/v2_ex.ckpt"
CKPT_HE = r"/home/nvidia/Desktop/3_WebDev/server/segmentation_weights/HE/v2_he.ckpt"
CKPT_MA = r"/home/nvidia/Desktop/3_WebDev/server/segmentation_weights/MA/v2_ma.ckpt"
CKPT_SE = r"/home/nvidia/Desktop/3_WebDev/server/segmentation_weights/SE/v2_se.ckpt"

# Creating the different models by also loading their weights
segmentation_model_EX = SegmentationModel.load_from_checkpoint(CKPT_EX).cuda()
segmentation_model_HE = SegmentationModel.load_from_checkpoint(CKPT_HE).cuda()
segmentation_model_MA = SegmentationModel.load_from_checkpoint(CKPT_MA).cuda()
segmentation_model_SE = SegmentationModel.load_from_checkpoint(CKPT_SE).cuda()
classification_model = ClassificationModel.load_from_checkpoint(CKPT_CLASSIFICATION).cuda()


def apply_mask(image, mask, color = [0.0, 0.0, 1.0], alpha=1):
    """
    Combining the image and the mask to create a new 
    image with the mask applied by applying different color
    The code is adapted from:
    https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/visualize.py#L85

    Args:
        image (PIL.Image): The image to be masked.
        mask (PIL.Image): The mask to be applied.
        color (list): The color to be applied to the mask.
        alpha (float): The alpha value used for color transparency

    Returns:
        PIL.Image: The image with the mask applied.
    """
    # Converting the image into an numpy array
    image_array = np.array(image)

    # Going through the image array and applying the mask
    for c in range(3):
        image_array[:, :, c] = np.where(mask == 1, 
                                  image_array[:, :, c] * (1 - alpha) + alpha * color[c] * 255, 
                                  image_array[:, :, c]
                                  )
        final_image = Image.fromarray(image_array)                       
    return final_image

def segmentation_single(model, image, color = [0.0, 0.0, 1.0]):
    """
    This method applies the model to a given image 
    and returns the new predicted image with the mask applied

    Args:
        model (torch.nn.Module): The model to be applied to the image
        image (PIL.Image): The image to be masked.
        color (list): The color to be applied to the mask (RGB).
    
    Returns:
        PIL.Image: The image with the mask applied.
    """
    cuda = torch.device('cuda:0')
    image_array = np.transpose(np.array(image), (2, 0, 1))
    x_tensor = torch.from_numpy(image_array).to(device=cuda)
    with torch.no_grad():
        model.eval().to(device=cuda)
        logits = model(x_tensor.to(device=cuda)).to(device=cuda)
        pr_masks = logits.sigmoid().squeeze().to(device=cuda)
    output_mask = (pr_masks>0.5).int().to(device=cuda)
    image_with_predicted_mask = apply_mask(image.copy(), output_mask.cpu().numpy(), color)
    return(image_with_predicted_mask)

def segmentation_ex(image, color = [0.0, 1.0, 1.0]):
    """
    This method applies a given segmentation model to a given image 
    for the Hard Exudate segmentation task.
    The image is also preprocessed by resizing it to 512x512.

    Args:
        image (PIL.Image): The image to be masked.
        color (list): The color to be applied to the mask (RGB).

    Returns:
        PIL.Image: The image with the mask applied for Hard Exudate.
     
    """

    preprocessed_image = pre_process_image(image, output_size=(512, 512))
    image_prediction = segmentation_single(segmentation_model_EX, preprocessed_image, color)
    return image_prediction

def segmentation_he(image, color = [0.0, 0.0, 1.0]):
    """
    This method applies a given segmentation model to a given image 
    for the Hemorrhages segmentation task.
    The image is also preprocessed by resizing it to 512x512.

    Args:
        image (PIL.Image): The image to be masked.
        color (list): The color to be applied to the mask (RGB).

    Returns:
        PIL.Image: The image with the mask applied for Hemorrhages.
     
    """
    preprocessed_image = pre_process_image(image, output_size=(512, 512))
    image_prediction = segmentation_single(segmentation_model_HE, preprocessed_image, color)
    return image_prediction

def segmentation_ma(image, color = [0.0, 0.0, 1.0]):
    """
    This method applies a given segmentation model to a given image 
    for the Microaneurysms segmentation task.
    The image is also preprocessed by resizing it to 512x512.

    Args:
        image (PIL.Image): The image to be masked.
        color (list): The color to be applied to the mask (RGB).

    Returns:
        PIL.Image: The image with the mask applied for Microaneurysms.
    """
    preprocessed_image = pre_process_image(image, output_size=(512, 512))
    image_prediction = segmentation_single(segmentation_model_MA, preprocessed_image, color)
    return image_prediction

def segmentation_se(image, color = [1.0, 1.0, 1.0]):
    """
    This method applies a given segmentation model to a given image 
    for the Soft Exudate segmentation task.
    The image is also preprocessed by resizing it to 512x512.

    Args:
        image (PIL.Image): The image to be masked.
        color (list): The color to be applied to the mask (RGB).

    Returns:
        PIL.Image: The image with the mask applied for Soft Exudate.
    """
    preprocessed_image = pre_process_image(image, output_size=(512, 512))
    image_prediction = segmentation_single(segmentation_model_SE, preprocessed_image, color)
    return image_prediction

def segmentation_final(image):
    """
    This method applies a given segmentation model to a given image for all the segmentation tasks.

    Args:
        image (PIL.Image): The image to be masked.
    
    Returns:
        PIL.Image: Four images with their segmentation masks applied for all the segmentation tasks.
    """
    image_ex = segmentation_ex(image)
    image_he = segmentation_he(image)
    image_ma = segmentation_ma(image)
    image_se = segmentation_se(image)
    return image_ex, image_he, image_ma, image_se

def classification(model, image):
    """
    This method applies a given classification model to a given image
    for predicting its severity degree.

    Args:
        model (torch.nn.Module): The model to be applied to the image
        image (PIL.Image): The image to be masked.
    
    Returns:
        pr_label (int): The predicted label for the image.
    """
    cuda = torch.device('cuda:0')
    image_array = np.transpose(np.array(image), (2, 0, 1))
    x_tensor = torch.from_numpy(image_array).to(device=cuda)
    with torch.no_grad():
        model.eval().to(device=cuda)
        logits = model(x_tensor.to(device=cuda)).to(device=cuda)
        pr_label = logits.argmax(1).squeeze().to(device=cuda)
    return pr_label.cpu().numpy()

def classification_final(image):
    """
    This method applies a given classification model to a given image
    for predicting its severity degree.
    By also applying preprocessing to the image.

    Args:
        image (PIL.Image): The image to be masked.
    Returns:
        label (int): The predicted label for the image.
    """
    preprocessed_image = pre_process_image(image, output_size=(512, 512))
    label = classification(classification_model, preprocessed_image)
    return label

# if __name__ == "__main__":
#     print("Hello, World!")
#     # print(torch.cuda.is_available())
#     # print(torch.device('cuda:0'))
#     image = Image.open(r"/home/nvidia/Desktop/0039_2.png")
#     preprocessed_image = pre_process_image(image, output_size=(512, 512))
#     print(preprocessed_image)
#     pr_label = classification(classification_model, preprocessed_image)
#     image_ex, image_he, image_ma, image_se  = segmentation_final(preprocessed_image)
#     print(pr_label)
#     print(image_ex)