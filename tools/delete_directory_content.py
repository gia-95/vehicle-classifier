#################################################################
# TopNetwork.it                                                 #
# Script: delete_directory_content.py                           #
# Date: 05/09/2025                                              #
# Author: Mauro Chiandone                                       #
# Usage: python delete_directory_content.py <directory_path>    #
# ###############################################################
# This script deletes all files and subdirectories              #
# located under given <directory_path>                          #
# ASKS for confirmation                                         #
#################################################################
import shutil
import sys,os
import glob

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'Script Usage:\n{sys.argv[0]} <directory_path>')
        sys.exit(f"Error: Wrong arguments for script '{sys.argv[0]}'")
    
    directory_path = sys.argv[1]
    
    # Check if the directory exists
    if os.path.isdir(directory_path):
        #"The directory exists."
        pass
    else:
        sys.exit(f"Error: Directory {directory_path} not found")
        

    print(f"WARNING!!! This will DELETE ALL '{directory_path}' content")
    answer = input("Are you sure? [Y/N]")
    if answer.lower() == 'y':
        for file_path in glob.glob(f"{directory_path}/*"):
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete files or symbolic links
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete directories and their contents
        print("ALL CONTENTS DELETED")
    else:
        print("**Abort** Nothing deleted")

