import os


def split_zip(file_path, chunk_size=1024 * 1024 * 100, delete_original_file=False):
    """
        1024 represents the number of bytes in a kilobyte (KB).
        1024 * 1024 calculates to 1,048,576 bytes, which is 1 megabyte (MB).
        1024 * 1024 * 100 therefore equals 100 MB (104,857,600 bytes).

    :param file_path: full file path of zip file
    :param chunk_size: to define the size of each chunk
    :param delete_original_file: option to remove original zip file
    :return:
    """
    with open(file_path, "rb") as f:
        i = 0
        while chunk := f.read(chunk_size):
            with open(f"{file_path}.part{i}", "wb") as chunk_file:
                chunk_file.write(chunk)
            i += 1

    if delete_original_file:
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("Splitting complete!")


