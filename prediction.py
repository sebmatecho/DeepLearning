
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

def predict_image(custom_image_path:str, 
                  model: torch.nn.Module, 
                  device: torch.device = device
                  ):


  custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)/355
  custom_image_transform = transforms.Compose([
      transforms.Resize(size = (64,64))
  ])
  custom_image_transformed = custom_image_transform(custom_image)
  model.eval()
  with torch.inference_mode(): 
    custom_image_pred =  model(custom_image_transformed.unsqueeze(dim=0)).to(device)

  custom_image_pred_probs = torch.softmax(custom_image_pred, dim = 1)
  custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim = 1).to('cpu')
  return class_names[custom_image_pred_label]

def pred_and_plot_image(model:torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None,
                        transform = None, 
                        device:torch.device = device): 
  """
  Makes a prediction on a target image with a trained model and plots the image and predictions
  """
  # Load in the image 
  target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

  # Divide the image pixel values by 255 to get them between [0,1]
  target_image = target_image/255.

  # Transofmr if necessary
  if transform: 
    target_image = transform(target_image)

  # Make sure the model is on the target device
  model.to(device)

  #Turn on eval/inference mode and make a prediction
  model.eval()
  with torch.inference_mode():
    target_image = target_image.unsqueeze(dim = 0)
    
    # Make a prediction on the image with an extra dimension
    target_image_pred = model(target_image.to(device)) # make sure the target image is in the right device
  
  # Convert logits -> prediction probabilities
  target_image_pred_probs = torch.softmax(target_image_pred, dim = 1)

  # Convert prediction probabilities -> predicted labels
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim = 1)

  # plot the image alongside the prediction and prediction probability 
  plt.imshow(target_image.squeeze().permute(1,2,0))
  if class_names: 
    title = f"pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.4f}"
  else: 
    title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.4f}"
  plt.title(title)
  plt.axis(False);
