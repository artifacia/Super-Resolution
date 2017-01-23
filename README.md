# Image Super-Resolution

This is a project on improving the resolution of images in an attempt to make them look more appealing. This project is an implementation of Super-Resolution CNN(SRCNN) which uses 2-dimensional convolutional neural networks for super resolution. In order to make the project suitable for on-device applications, it is made sure that the used neural networks are small in size and don't require huge computations.

### Prerequisites
You will require the following dependencies to get the code running.
* Python 2.7
* Numpy
* Tensorflow
* cv2

### Training your own model

Clone the repository and store low-resolution images and their corresponding high-resolution images in folders Data/ and Labels/ respectively inside a desired folder, pass this path as input_path to SRCNN. Run SRCNN_Train_OS.py to start training, tweak hyperparameters like learning rate and number of epochs in the same file.

### Sample images from our test runs

Input low resolution image - 1

![Input low resolution](https://cloud.githubusercontent.com/assets/10481640/22190209/abf98d34-e147-11e6-8e4c-11f34871c03a.jpg "Input low resolution image - 1")

Output high resolution image - 1

![Output high resolution](https://cloud.githubusercontent.com/assets/10481640/22190197/9ae1d416-e147-11e6-9201-1cf485b239e5.jpg "Output high resolution image - 1")

Input low resolution image - 2 ---> Output high resolution image - 2

![Input low resolution](https://cloud.githubusercontent.com/assets/10481640/22190219/b76f3f56-e147-11e6-9373-9a7ba04308f7.jpg "Input low resolution image - 2") ![Output high resolution](https://cloud.githubusercontent.com/assets/10481640/22190214/b2455e34-e147-11e6-9f62-04d2ca802873.jpg "Output high resolution image - 2")

## Authors

* **Vijay Veerabadran** - *Artifacia Pvt Ltd* - [vijayvee](https://github.com/vijayvee)
