import os
import shutil


def copy_folder(source, destination):
    try:
        # Check if the source folder exists
        if not os.path.exists(source):
            print("Source folder does not exist.")
            return

        # Remove the destination folder if it already exists
        if os.path.exists(destination):
            shutil.rmtree(destination)

        # Copy the folder and its contents recursively
        shutil.copytree(source, destination)

        print("Folder and contents copied successfully.")
    except Exception as e:
        print("An error occurred while copying the folder:", str(e))


def get_folder_path_input(prompt, default=None):
    while True:
        path = input(prompt)
        if not path:
            if default:
                return default
            print("Invalid input. Please try again.")
        elif not os.path.exists(path):
            print("Path does not exist. Please try again.")
        else:
            return path


def main():
    # Prompt the user for the source folder path or use a default path
    default_source = "/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data"
    use_default_source = input("Use default source folder? (Y/N): ").upper() == "Y"
    if use_default_source:
        source_folder = default_source
    else:
        source_folder = get_folder_path_input("Enter the source folder path: ")

    # Prompt the user for the destination folder path or use a default path
    default_destination = "/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/utils/buckupData"
    use_default_destination = input("Use default destination folder? (Y/N): ").upper() == "Y"
    if use_default_destination:
        destination_folder = default_destination
    else:
        destination_folder = get_folder_path_input("Enter the destination folder path: ")

    # Copy the folder and its contents
    copy_folder(source_folder, destination_folder)


if __name__ == "__main__":
    main()
