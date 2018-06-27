from os import listdir
__all__= [name[:-3]
    for name in listdir('./clustering') if name[0] != '_']
