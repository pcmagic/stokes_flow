import subprocess
import os

devnull = open(os.devnull, 'w')
latex_installed = not subprocess.call(['which', 'latex'],
                                      stdout=devnull, stderr=devnull)
if latex_installed:
    print('wget installed!')
else:
    print('wget missing in path!')
