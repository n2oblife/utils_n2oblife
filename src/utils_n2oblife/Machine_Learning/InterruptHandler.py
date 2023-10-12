import torch
import sys

def keybInterrupt(model, model_name, optimizer,
                  num_epochs, training, dataset,
                  model_dir, 
                  val_losses, val_acc, 
                  train_losses, train_acc=None):
    prompt = input("\nKeyboard interrupt, do you want to the model and its metrics ? \nyes/no \n")
    if (prompt=='yes' or prompt=='y' or prompt=='Y'):
        if train_acc == None :
            model_state = {'model name': model_name,
                'model': model,
                'optimizer': optimizer,
                'epoch': num_epochs,
                'training': training,
                'dataset': dataset,
                'metrics' : {'train_loss' : train_losses,
                            'val_loss' :val_losses,
                            'val_acc' : val_acc}
                }
        else :
            model_state = {'model name': model_name,
                'model': model,
                'optimizer': optimizer,
                'epoch': num_epochs,
                'training': training,
                'dataset': dataset,
                'metrics' : {'train_loss' : train_losses,
                            'train_acc' : train_acc,
                            'val_loss' :val_losses,
                            'val_acc' : val_acc}
                }
        torch.save(model_state, model_dir)
        print("Model saved in the dir : ",model_dir)
    else :
        print("Model not saved")
    return None