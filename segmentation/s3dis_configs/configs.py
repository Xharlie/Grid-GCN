import yaml
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils.utils import get_timestamp

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs.yaml'))as f:
    configs = yaml.load(f, Loader = yaml.FullLoader)
configs['num_channel'] = 6 if configs['use_normal'] else 3
# configs['git_hash'] = get_git_hash()
configs['timestamp'] = get_timestamp()


if __name__ == "__main__":
    print(configs)