import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Doit retourner True
print(torch.cuda.device_count())  # Doit être > 0
print(torch.cuda.get_device_name(0))  # Doit afficher "RTX 3070 Ti"
