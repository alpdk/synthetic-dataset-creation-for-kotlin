import os
import sys
import shutil
import requests, zipfile, io


# Install zip archive and unpack it
def install_arc_and_unpack(link, cur_dir):
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(cur_dir)


# Install kotlinc by link with checks
def install_kotlinc(link, cur_dir, kotlinc_dir):
    # Checking kotlinc existence
    if os.path.exists(kotlinc_dir):
        command = input("Kotlinc does exist. Do you want to update it?(y/N):")

        command = command.lower()

        if command == 'y':
            try:
                shutil.rmtree(kotlinc_dir)

                print("Kotlinc uninstalled successfully. Start installing...", end='\n')
                install_arc_and_unpack(link, cur_dir)
            except OSError as o:
                print(f"Error, {o.strerror}: /usr/local/kotlinc", end='\n')
    else:
        print("Kotlinc does not exist. Start installing...", end='\n')
        install_arc_and_unpack(link, cur_dir)


def setup_kotlinc(link='https://github.com/JetBrains/kotlin/releases/download/v2.0.20/kotlin-compiler-2.0.20.zip'):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    kotlinc_dir = os.path.join(cur_dir, "kotlinc")

    install_kotlinc(link, cur_dir, kotlinc_dir)

    path_to_kotlinc = os.path.join(kotlinc_dir, "bin")
    current_path = os.environ.get('PATH')
    path_list = current_path.split(os.pathsep)

    if path_to_kotlinc in path_list:
        print(f"The path '{path_to_kotlinc}' ALREADY in the PATH.")
    else:
        os.environ['PATH'] += ":" + path_to_kotlinc
        print(f"The path '{path_to_kotlinc}' ADDED to the PATH.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        setup_kotlinc()
    else:
        setup_kotlinc(sys.argv[1])