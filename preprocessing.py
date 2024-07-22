import os


def run():
    folders = [
        'data',
        'images',
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    run()