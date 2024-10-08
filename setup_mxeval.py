import os
import sys
import shutil

from git import Repo


def install_mxeval(link='https://github.com/amazon-science/mxeval.git'):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    mxeval_dir = os.path.join(cur_dir, "mxeval")

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


def setup_mxeval(mxeval_parent_dir=os.path.dirname(os.path.realpath(__file__))):
    mxeval_dir = os.path.join(mxeval_parent_dir, 'mxeval')

    if os.path.isdir(mxeval_dir):

        if mxeval_dir in sys.path:
            sys.path.remove(mxeval_dir)

        sys.path.insert(0, mxeval_dir)
    else:
        install_mxeval()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        setup_mxeval()
    elif len(sys.argv) == 2:
        setup_mxeval(sys.argv[1])
    else:
        print("Wrong number of arguments")
