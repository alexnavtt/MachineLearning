import struct
import numpy as np

def getNextField(data, format, idx):
    datasize = struct.calcsize(format)
    (observed_data,) = struct.unpack(format, data[idx : idx + datasize])
    idx += datasize
    return (observed_data, idx)
    

def readMNistLabels(filepath, magic_number):
    header_fields = [">i",  # 32-bit int : magic number
                     ">i"]  # 32-bit int : number of labels


    # Get the data from the file
    data = open(filepath, "rb").read()

    # The current position in the binary file
    idx = 0 

    # Get the magic number and make sure it matches
    (observed_magic_number, idx) = getNextField(data, header_fields[0], idx)
    if observed_magic_number != magic_number:
        print("Something's wrong, the magic number should have been {} but was instead {}".format(magic_number, observed_magic_number))

    # Determine how many images are in the dataset
    (image_count, idx) = getNextField(data, header_fields[1], idx)

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

    # Get the magic number and make sure it matches
    (observed_magic_number, idx) = getNextField(data, header_fields[0], idx)
    if observed_magic_number != magic_number:
        print("Something's wrong, the magic number should have been {} but was instead {}".format(magic_number, observed_magic_number))

    # Determine how many images are in the dataset
    (image_count,  idx) = getNextField(data, header_fields[1], idx)
    (row_count,    idx) = getNextField(data, header_fields[2], idx)
    (column_count, idx) = getNextField(data, header_fields[3], idx)
    print("Retrieved {} images with {} rows and {} columns".format(image_count, row_count, column_count))

    # Get the image data
    images = []
    image_format = ">" + "B"*row_count*column_count
    datasize = struct.calcsize(image_format)

    for _ in range(image_count):
        # Get the image data
        image_pixels = struct.unpack(image_format, data[idx : idx + datasize])
        idx += datasize

        # Convert it to a numpy MxN matrix
        new_image = np.zeros((row_count, column_count), np.uint8)
        for i in range(column_count):
            for j in range(row_count):
                new_image[i,j] = 255 - image_pixels[i*row_count + j]

        images.append(new_image)

    return images
    