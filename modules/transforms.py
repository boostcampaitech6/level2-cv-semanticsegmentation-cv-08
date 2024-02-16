import albumentations as A

def get_transform_function(transform_function_str,config):
    
    if transform_function_str == "baseTransform":
        return baseTransform(config)
    

def baseTransform(config):
    return A.Compose([
        A.Resize(config['input_size'], config['input_size']),
    ])