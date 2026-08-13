import torchvision.transforms as T

def get_transforms():
    
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5), # تقليب أفقي
        T.RandomVerticalFlip(p=0.5),   # تقليب رأسي (مناسب جداً للصخور)
        T.RandomRotation(degrees=45),  # دوران للصور
        T.ColorJitter(brightness=0.1, contrast=0.1), # تغيير خفيف للإضاءة والتباين
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform