import os
import sys

# Add mxeval_dir to sys.path for usages
def setup_mxeval(mxeval_parent_dir):
    # Get the path to the mxeval directory
    mxeval_dir = os.path.join(mxeval_parent_dir, 'mxeval')

    # Add mxeval to sys.path
    if sys.path[0] != mxeval_dir:

        if mxeval_dir in sys.path:
            sys.path.remove(mxeval_dir)

        sys.path.insert(0, mxeval_dir)

    # Print all system path for packages
    # sys.path.remove(mxeval_dir)
    # print(sys.path, sep='\n')
