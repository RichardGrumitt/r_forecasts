import glob
from natsort import natsorted
import os


def mk_file_list(directory, suffix, absolute=False):

    '''
    Function for creating a text file listing all files in a given directory, with some suffix.
    This sorts the files so they are in ascending numerical order.

    Inputs
    ------
    directory - The directory where the map files are located.
    suffix - The suffix of the files you are interested in.
    absolute (optional) - Set top True if you want the absolute file paths to be listed.

    Outputs
    -------
    file_list - A list of the map files in the specified directory.

    '''

    file_list = natsorted(glob.glob('{}*{}'.format(directory, suffix)))
    
    if absolute == True:
        
        temp_list = []
        
        for path in file_list:
            
            temp_list.append(os.path.abspath(path))
            
        file_list = temp_list

    return file_list
