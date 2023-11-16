from model.movenet import movenet_process

def main():
    test_image = 'model/goddess.jpg'
    movenet_process(test_image)

if __name__ == "__main__":
    main()