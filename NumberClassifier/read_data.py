import struct
import numpy as np

def readMNistLabels(filepath, magic_number):
    header_fields = [">i",  # 32-bit int : magic number
                     ">i"]  # 32-bit int : number of labels

    idx = 0 # The current position in the binary file

    # Get the data from the file
    data = open(filepath, "rb").read()

    # Start unpacking the data
    datasize = struct.calcsize(header_fields[0])

    # Get the magic number and make sure it matches
    (observed_magic_number,) = struct.unpack(">i", data[idx : idx+datasize])
    if observed_magic_number != magic_number:
        print("Something's wrong, the magic number should have been {} but was instead {}".format(magic_number, observed_magic_number))
    idx += datasize

    # Determine how many images are in the dataset
    datasize = struct.calcsize(header_fields[1])
    (image_count,) = struct.unpack(">i", data[idx : idx+datasize])
    print("Opened a file with {} images".format(image_count))
    idx += datasize

    # Get the labels for each image
    datasize = struct.calcsize(">" + "B"*image_count)
    labels = struct.unpack(">" + "B"*image_count, data[idx : idx + datasize])
    return labels


def readMNistImages(filepath, magic_number):
    # Define the format of the file
    header_fields = [">i",  # 32-bit int : magic number
                     ">i",  # 32-bit int : number of images
                     ">i",  # 32-bit int : number of rows per image
                     ">i"]  # 32-bit int : number of colums per image

    idx = 0 # The current position in the binary file

    # Get the data from the file
    data = open(filepath, "rb").read()

    # Start unpacking the data
    datasize = struct.calcsize(header_fields[0])

    # Get the magic number and make sure it matches
    (observed_magic_number,) = struct.unpack(">i", data[idx : idx+datasize])
    if observed_magic_number != magic_number:
        print("Something's wrong, the magic number should have been {} but was instead {}".format(magic_number, observed_magic_number))
    idx += datasize

    # Determine how many images are in the dataset
    datasize = struct.calcsize(header_fields[1])
    (image_count,) = struct.unpack(">i", data[idx : idx+datasize])
    print("Opened a file with {} images".format(image_count))
    idx += datasize

    # Get the row and column count for each image
    datasize = struct.calcsize(header_fields[2])
    (row_count,) = struct.unpack(header_fields[2], data[idx : idx+datasize])
    idx += datasize

    datasize = struct.calcsize(header_fields[3])
    (column_count,) = struct.unpack(header_fields[3], data[idx : idx + datasize])
    idx += datasize
    print("Each image has {} rows and {} columns".format(row_count, column_count))

    # Get the image data
    images = []
    image_format = ">" + "B"*row_count*column_count
    datasize = struct.calcsize(image_format)
    for _ in range(image_count):
        new_image = np.zeros((row_count, column_count), np.uint8)
        image_pixels = struct.unpack(image_format, data[idx : idx + datasize])
        idx += datasize

        for i in range(column_count):
            for j in range(row_count):
                new_image[i,j] = 255 - image_pixels[i*row_count + j]

        images.append(new_image)

    return images
    