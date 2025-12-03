import torch
from PIL import Image
from src.utils.transforms import get_transforms
from src.models.efficientnet import EfficientNetB3

class Predictor:
    def __init__(self, model_path, config, device):
        self.device = device
        self.config = config
        self.classes = config['classes']
        
        # Load Model
        self.model = EfficientNetB3(num_classes=config['num_classes'])
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Transforms
        _, self.transform = get_transforms(config['img_size'])
        
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        class_name = self.classes[predicted.item()]
        conf_score = confidence.item()
        
        return class_name, conf_score
