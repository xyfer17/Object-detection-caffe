/**
 * @file        Inference.cpp
 * @brief       This file contains functions for object detection model implementation.
 *
 * @author      Naveen Kumar Yadav
 * @bugs        No known bugs.

 All rights reserved by xyfer17

*/


/*
 *#####################################################################
 *  Initialization block
 *  ---------------------
 *#####################################################################
 */

/* --- Standard Includes --- */

#include<algorithm>
#include<iosfwd>
#include<memory>
#include<string>
#include<vector>
#include<iostream>
#include<utility>

/* --- Project Includes --- */

#include<caffe/caffe.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



const float CONF_THRESH = 0.35;
const float NMS_THRESH = 0.45;

using namespace caffe;
using namespace std;

/*
 *#####################################################################
 *  Process block
 *  ---------------------
 *#####################################################################
 */




 /**
  * @brief
  *
  * @param      image  input image file for detection
  * @param      pred_boxes  boxes co-ordinates for detection
  * @param      confidence  score of the model for object
  * @param      labels   label of the output
  * @param      h   height of the image orientation
  * @param      w   width of the image orientation
  *
  * @return     vis.jpg image file
  */


void vis_detections(cv::Mat image, vector<vector<float> > pred_boxes, vector<float> confidence,vector<int> labels, int h , int w) // function definition
{
	int lab;
	string labs[91] = { "background" , "person" , "bicycle" , "car" , "motorcycle" ,
     "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light",
     "fire hydrant", "N/A" , "stop sign", "parking meter", "bench" ,
     "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" ,
     "bear" , "zebra" , "giraffe" , "N/A" , "backpack" , "umbrella" ,
     "N/A" , "N/A" , "handbag" , "tie" , "suitcase" , "frisbee" , "skis" ,
     "snowboard" , "sports ball", "kite" , "baseball bat", "baseball glove",
     "skateboard" , "surfboard" , "tennis racket", "bottle" , "N/A" ,
     "wine glass", "cup" , "fork" , "knife" , "spoon" , "bowl" , "banana" ,
     "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog",
     "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant",
     "bed" , "N/A" , "dining table", "N/A" , "N/A" , "toilet" , "N/A" ,
     "tv" , "laptop" , "mouse" , "remote" , "keyboard" , "cell phone",
     "microwave" , "oven" , "toaster" , "sink" , "refrigerator" , "N/A" ,
     "book" , "clock" , "vase" , "scissors" , "teddy bear", "hair drier",
     "toothbrush"};

	for(int i=0; i<pred_boxes.size();i++){
		lab = labels[i] ;
// for putting rectangle around the detection
  if (confidence[i] > CONF_THRESH){
cv::rectangle(image, cv::Point(pred_boxes[i][0] * w, pred_boxes[i][1] * h), cv::Point(pred_boxes[i][2] * w, pred_boxes[i][3] * h), cv::Scalar(0, 255, 0), 2, 8, 0);
// for putting text on the detection
cv::putText(image, labs[lab] ,cv::Point(pred_boxes[i][0] * w, pred_boxes[i][1] * h - 4),cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0),2);
	} }
}



int main(int argc, char *argv[])
{

  string model_file = "model/deploy.prototxt";
	string weights_file = "model/deploy.caffemodel";


Caffe::set_mode(Caffe::CPU);
 Net<float> *net_ = new Net<float>(model_file,TEST);
 net_->CopyTrainedLayersFrom(weights_file);



  cv::Mat cv_resized;
  // reading image file
	cv::Mat cv_img = cv::imread("img/3.jpg");
  // image resize
  cv::resize(cv_img, cv_resized, cv::Size(300, 300));

	cv::Mat cv_new(cv_resized.rows, cv_resized.cols, CV_32FC3, cv::Scalar(0, 0, 0));
	if (cv_img.empty())
	{
		std::cout << "Can not get the image file !" << endl;

	}



	int height = int(cv_resized.rows );
	int width = int(cv_resized.cols );




	float *data_buf= new float[height * width * 3];



	// image preprocess

	for (int h = 0; h < cv_resized.rows; ++h)
	{
		for (int w = 0; w < cv_resized.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(127.5);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(127.5);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(127.5);

		}
	}



        for (int h = 0; h < cv_resized.rows; ++h)
	{
		for (int w = 0; w < cv_resized.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[0]) / float(127.5);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[1]) / float(127.5);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[2]) / float(127.5);

		}
	}







	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(2 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(0 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(1 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}





	const float* detections = NULL;
        vector<int> labels;
        vector<float> conf;
        vector<vector<float>> boxes;

	net_->blob_by_name("data")->Reshape(1, 3, height, width);
        //Reshape data layer according to the image
	net_->blob_by_name("data")->set_cpu_data(data_buf);
        //Pass image through the network.
	net_->ForwardFrom(0); //Inference model
	// Collect the output of the network from it"s end node in a variable called detections
        detections = net_->blob_by_name("detection_out")->cpu_data();


      // image postprocess
      while(1) {

	if (*detections == 0)
	   detections++;

	else if (*detections !=0 && *detections <= 0.001)
		break;

	else
	{
	vector<float> v1;
    labels.push_back(*detections);
	detections++;
	conf.push_back(*detections);
	detections++;
	v1.push_back(*detections);
	detections++;
	v1.push_back(*detections);
	detections++;
	v1.push_back(*detections);
	detections++;
	v1.push_back(*detections);
    boxes.push_back(v1);
	detections++;
	}
 }



vis_detections(cv_img, boxes, conf, labels ,cv_img.rows, cv_img.cols); // call for visualization
cv::imwrite("vis.jpg",  cv_img);

std::cout << "finished" << endl;

return 0;

}
