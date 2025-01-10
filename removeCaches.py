import os, shutil

# walk through the current directory recursively and remove all __pycache__ directories
def removeCaches():
    for root, dirs, _ in os.walk("."):
        for dir in dirs:
            if dir == "__pycache__":
                shutil.rmtree(os.path.join(root, dir))

if __name__ == "__main__":
    removeCaches() 
    