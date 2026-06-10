import os

def load_model(model_path, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.load(model_path)
    model.to(device)
    model.eval()

    print("Custom model loaded successfully!")
    return model, device

def load_clf_model(model_name, num_classes=2, IMAGE_SIZE = (224, 224)):
    num_classes = num_classes
    IMAGE_SIZE = IMAGE_SIZE

    current_dir = os.path.dirname(__file__)

    model_path = os.path.join(current_dir,"model", "weights", model_name)


    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    model, device = load_model(model_path,num_classes)

    return model, device

def load_seg_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1, feature_dims=64) 
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Custom model loaded successfully!")
    return model, device

