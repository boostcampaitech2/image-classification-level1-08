import subprocess


def main():
    subprocess.call(['python3', 'train.py', '-c', "config.json"])
    subprocess.call(['python3', 'infer.py'])
    
if __name__ == '__main__':
    main()
