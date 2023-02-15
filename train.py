"""
Trains a PyTorch image classification model using device-agnostic code
"""
import os
import torch 
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--LearningRate', metavar = 'LEARNING_RATE', type = float)
parser.add_argument('--Epochs', metavar = 'NUM_EPOCHS', type = int)
parser.add_argument('--BatchSize', metavar = 'NUM_EPOCHS', type = int)
parser.add_argument('--HiddenUnits', metavar = 'NUM_EPOCHS', type = int)
args = parser.parse_args()


# Setup hyperparameters
NUM_EPOCHS = args.Epochs
BATCH_SIZE = args.BatchSize
HIDDEN_UNITS = args.HiddenUnits
LEARNING_RATE = args.LearningRate

# Setup directories
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize(size = (64,64)), 
  transforms.ToTensor()
])

# Create Dataloader's and get class_names
train_dataloader, test_dataloader, class_names  = data_setup.create_dataloaders(train_dir = train_dir,
                                                                                test_dir = test_dir, 
                                                                                transform  = data_transform, 
                                                                                batch_size = BATCH_SIZE)
# Create the model 
model = model_builder.TinyVGG(input_shape = 3, 
                              hidden_units = HIDDEN_UNITS, 
                              output_shape = len(class_names)).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), 
                            lr= LEARNING_RATE)

# Start the timer

start_time = timer()

# Start training with help from engine
engine.train(model = model, 
            train_dataloader = train_dataloader, 
            test_dataloader = test_dataloader, 
            loss_fn = loss_fn, 
            optimizer = optimizer,
            epochs = NUM_EPOCHS, 
            device = device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model to file 
utils.save_model(model = model, 
                target_dir = 'models', 
                model_name = '05_going_modular_script_mode_tinyvgg_model.pth')
