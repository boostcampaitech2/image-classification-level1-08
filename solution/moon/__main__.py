import train
import json
import argparse

def run(config):
    train.train(config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    
    args = parser.parse_args()
    with open(args.config,'r') as f:
        c = json.load(f)
    run(c)
