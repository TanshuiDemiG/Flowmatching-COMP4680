import json
import os

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "log.txt")
        
    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

def save_config(config, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)
