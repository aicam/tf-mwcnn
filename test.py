import os

mode = 0o666
try :
    os.mkdir('results', mode)
except FileExistsError:
    pass