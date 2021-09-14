from matplotlib import pyplot as plt

# Note : Data recorded by hand from main file

def main():
    test_count = 400
    data_count    = [ 10,  50, 100, 150, 200, 250, 300, 350, 400]
    success_count = [139, 242, 276, 299, 302, 309, 316, 319, 325]

    success_count = [x*100/test_count for x in success_count]

    plt.plot(data_count, success_count)
    plt.xlabel("Training Image Count")
    plt.ylabel("Success rate (%)")
    plt.show()

    data_count    = [  5,  10,  15,  20,  25,  30,  40,  50,  75, 100]
    success_count = [230, 297, 318, 323, 321, 321, 324, 323, 324, 324]

    success_count = [x*100/test_count for x in success_count]

    plt.plot(data_count, success_count)
    plt.xlabel("Eigenvector Count")
    plt.ylabel("Success rate (%)")
    plt.show()

if __name__ == "__main__":
    main()