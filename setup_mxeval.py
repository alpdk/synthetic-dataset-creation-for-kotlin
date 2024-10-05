import os
import sys
import shutil

from git import Repo
from get_cur_dir import get_cur_dir


# Install mxeval by link
def install_mxeval(link='https://github.com/amazon-science/mxeval.git'):
    cur_dir = get_cur_dir()
    mxeval_dir = os.path.join(cur_dir, "mxeval")

    # Check dir for existence
    if os.path.isdir(mxeval_dir):
        command = input("Directory does exist. Do you want to update it?(y/N):")

        command = command.lower()

        if command == 'y':
            try:
                shutil.rmtree(mxeval_dir)

                print("Directory removed successfully. Start cloning...", end='\n')
                Repo.clone_from(link, mxeval_dir)
            except OSError as o:
                print(f"Error, {o.strerror}: {mxeval_dir}", end='\n')
    else:
        print("Directory does not exist. Start cloning...", end='\n')
        Repo.clone_from(link, mxeval_dir)


# Add mxeval_dir to sys.path for usages
def setup_mxeval(mxeval_parent_dir='/home/alpdk/gitRepos/synthetic-dataset-creation-for-kotlin/'):
    # Get the path to the mxeval directory
    mxeval_dir = os.path.join(mxeval_parent_dir, 'mxeval')

    # Add mxeval to sys.path
    if sys.path[0] != mxeval_dir:

        if mxeval_dir in sys.path:
            sys.path.remove(mxeval_dir)

        sys.path.insert(0, mxeval_dir)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        setup_mxeval()
    else:
        setup_mxeval(sys.argv[1])
