 # Object-detection-caffe








 ### Steps to build Caffe-SSD for object detection:

 #### Step 1:

 **Note: Please make sure that you have Ubuntu 16.04 as the OS environment before running this module**

 * Check that you have gcc version 6 or greater which is supported by caffe. Please follow the steps mentioned over [here](https://gist.github.com/zuyu/7d5682a5c75282c596449758d21db5ed) to install **gcc-6**.
 * Install all dependencies:

 ```shell
 $ sudo add-apt-repository ppa:lkoppel/opencv

 $ sudo apt-get update

 $ sudo apt install libopencv-calib3d3.2 libopencv-core3.2 libopencv-features2d3.2  libopencv-flann3.2 libopencv-highgui3.2 libopencv-imgcodecs3.2 libopencv-imgproc3.2 libopencv-ml3.2 libopencv-objdetect3.2 libopencv-photo3.2 libopencv-shape3.2 libopencv-stitching3.2 libopencv-superres3.2 libopencv-video3.2 libopencv-videoio3.2 libopencv-videostab3.2 libopencv-viz3.2 libopencv3.2-jni libopencv3.2-java libopencv-contrib3.2

 $ sudo apt install libopencv-calib3d-dev libopencv-core-dev libopencv-features2d-dev  libopencv-flann-dev libopencv-highgui-dev libopencv-imgcodecs-dev libopencv-imgproc-dev libopencv-ml-dev libopencv-objdetect-dev libopencv-photo-dev libopencv-shape-dev libopencv-stitching-dev libopencv-superres-dev libopencv-video-dev libopencv-videoio-dev libopencv-videostab-dev libopencv-viz-dev libopencv-contrib-dev libopencv-dev
 $ sudo apt-get install libopencv-dev python3-opencv
 $ sudo apt-get install libopenblas-dev
 $ sudo apt-get install libboost-all-dev
 $ sudo apt install python3-pip
 $ pip3 install protobuf==3.11.3
 $ sudo apt-get install libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev
 $ sudo apt-get install python3-dev python-dev
 $ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
 ```

 * Build **boost library** from source.

 ```shell
 $ wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz

 $ tar xzvf boost_1_66_0.tar.gz
 $ cd boost_1_66_0/

 $ sudo apt-get update
 $ sudo apt-get install build-essential autotools-dev libicu-dev libbz2-dev
 $ ./bootstrap.sh --prefix=/usr/local/
 $ ./b2 -j8 variant=release link=shared threading=multi runtime-link=shared
 $ sudo ./b2 -j8 install
 ```

 * Build protobuf library from source.Download **protobuf-all-version-3.11.4.tar.gz** from [here](https://github.com/protocolbuffers/protobuf/releases/tag/v3.11.4), extract file and build it using below commands.

 ```shell
 $ sudo apt-get install autoconf automake libtool curl make g++ unzip cpio
 cd protobuf-3.11.4
 ./configure
 make
 make check
 sudo make install
 sudo ldconfig
 ```

 #### Step 2:

 * For ssdlite I have used python 2.7, because python 3 had issues. Ubuntu's deafult python would do the task or make sure you have python 2.7 on your system.

 ```
 git clone https://github.com/xyfer17/ssd-caffe.git
 cd ssd-caffe

 ```
 * Follow the below  command to build the ssd-caffe

 ```
 make all
 make test
 make runtest
 make pycaffe
 ```

  ####  Inference  process

 * clone this repository using the below command

 ```
 $ git clone https://github.com/xyfer17/Object-detection-caffe.git

  $ cd Object-detection-caffe/src

 ```

 * Follow this step to add your image file which you want to test using the detection model.

* open `inference.cpp` file using your text-editor and the search the below code inside it.
```
cv::Mat cv_img = cv::imread("img/3.jpeg")
```
* replace `img/3.jpeg` to your image file path .

* After this follow these command for execution process
 ```
  $ g++ -o app inference.cpp   -I /****/ssd-caffe/include/ -I /***/ssd-caffe/.build_release/src/  `pkg-config --libs --cflags opencv` -L /****/ssd-caffe/build/lib/ -lcaffe -lglog -lboost_system -lprotobuf -DCPU_ONLY=1

 ```
 ```
 $ export LD_LIBRARY_PATH=/path-to/ssd-caffe/build/lib/:$LD_LIBRARY_PATH

 $ export LD_LIBRARY_PATH=/path-to/boost_1_66_0/stage/lib/:$LD_LIBRARY_PATH

 $ ./app
 ```
