import gzip
import shutil
import os

def extract_and_delete_gz(file_path):
    """
    Extracts a .csv.gz file and deletes the original archive.

    Args:
        file_path (str): Full path to the .csv.gz file.

    Returns:
        str: Path to the extracted .csv file, or None if an error occurred.
    """
    if not file_path.endswith(".gz"):
        print(f"Skipping non-gzip file: {file_path}")
        return None

    try:
        # Determine the output file name
        output_file = file_path[:-3]  # Remove the ".gz" extension

        # Extract the .gz file
        with gzip.open(file_path, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Delete the original .gz file
        os.remove(file_path)
        print(f"Extracted and deleted archive: {file_path}")

        return output_file
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None



def extract_all_gz_in_dir(directory):
    """
    Extracts all .csv.gz files in the specified directory and deletes the archives.

    Args:
        directory (str): Path to the directory containing .csv.gz files.

    Returns:
        None
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                file_path = os.path.join(root, file)
                extract_and_delete_gz(file_path)
