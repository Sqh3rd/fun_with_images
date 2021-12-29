import os

for i, dir in enumerate(os.listdir('./input_images')):

    DIR = dir
    print(dir)
    if i < 
        continue
    FILES = []
    for f in os.listdir(f'./input_images/{DIR}'):
        if not f[:2] in FILES:
            FILES.append(f[:2])
    FILETYPE = '.jpg'

    PATH = f'./input_images/{DIR}/'

    for a, file in enumerate(FILES):
        if file[-1] != '_':
            file += '_'
        path = f'{PATH}{file}'
        
        print(file)

        for i in range(1,101):
            number = f'{i}'
            try:
                os.rename(f'{path}{number}{FILETYPE}', f'{PATH}{a%70}_{i}{FILETYPE}')
            except:
                continue