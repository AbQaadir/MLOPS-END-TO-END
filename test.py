# my_os_package.py

import os

print(os.getcwd())

def create_directory(directory_name):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created.")
    else:
        print(f"Directory '{directory_name}' already exists.")

def list_files(directory):
    """List files in a directory"""
    files = os.listdir(directory)
    if files:
        print(f"Files in directory '{directory}':")
        for file in files:
            print(file)
    else:
        print(f"No files found in directory '{directory}'.")

def delete_file(file_path):
    """Delete a file"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' deleted.")
    else:
        print(f"File '{file_path}' does not exist.")

def main():
    # Test the functions
    directory_name = "example_directory"
    create_directory(directory_name)

    file_path = os.path.join(directory_name, "example_file1.txt")
    with open(file_path, "w") as file:
        file.write("This is an example file.")

    list_files(directory_name)
    # delete_file(file_path)

if __name__ == "__main__":
    main()


# Path: test.py