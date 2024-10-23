import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input file")
    parser.add_argument('-d', '--dir', help="output directory")
    
    args = parser.parse_args()
    input = args.input
    dir = args.dir
    print("input:", input)
    print("dir", dir)