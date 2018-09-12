from os import listdir
import sys
from pathlib import Path
directory = Path(sys.argv[0]).parent / 'clustering'
__all__= [name[:-3]
    for name in listdir(str(directory)) if name[0] != '_']
