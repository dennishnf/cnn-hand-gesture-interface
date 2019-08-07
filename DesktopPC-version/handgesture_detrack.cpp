#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sstream>
 
using namespace cv;
using namespace std;

Mat img;
vector<Rect> hands;
CascadeClassifier Hand;


using namespace cv::dnn;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
  Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
  Point classNumber;
  
  minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
  *classId = classNumber.x;
}


 
int main(int argc, char **argv)
{
  VideoCapture cap(0);

  Hand.load("hand.xml");
  Rect2d bbox;
  int xx1,yy1,xx2,yy2;



  //! [Neural Netowork Preparation]
  CV_TRACE_FUNCTION();

  String modelTxt = "model/model_deploy.prototxt";
  String modelBin = "model/train_iter_2000.caffemodel";
  
  Net net;
  try {
    //! [Read and initialize network]
    net = dnn::readNetFromCaffe(modelTxt, modelBin);
    //! [Read and initialize network]
  }
  catch (cv::Exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    //! [Check that network was read successfully]
    if (net.empty()){
      std::cerr << "Can't load network by using the following files: " << std::endl;
      std::cerr << "prototxt:   " << modelTxt << std::endl;
      std::cerr << "caffemodel: " << modelBin << std::endl;
      exit(-1);
    }
    //! [Check that network was read successfully]
  }
  //! [Neural Netowork Preparation]
  
  int classId=0;
   



  while (true)
  {
    
    while(true)
    {
      cv::TickMeter t;
      t.start();
      cap>>img;
      Hand.detectMultiScale(img,hands,1.1,3,0|CV_HAAR_FIND_BIGGEST_OBJECT,Size(30,30));
      for(size_t  i=0; i < hands.size() ; i++)
      {
        int point1x = hands[i].x-0.9*hands[i].width;
        int point1y = hands[i].y+0.4*hands[i].width;
        int lengthxx = hands[i].width+2*0.9*hands[i].width;
        int lengthyy = hands[i].width+0.3*hands[i].width;
        rectangle(img, Point(point1x,point1y), Point(point1x+lengthxx,point1y+lengthyy), Scalar(0,0,255), 2, 1);
        Rect2d bbox0(Point(point1x,point1y), Point(point1x+lengthxx,point1y+lengthyy));
        bbox=bbox0;
      }
      putText(img, "Hand detection step", Point(30,35), FONT_HERSHEY_SIMPLEX , 0.9, Scalar(0,0,255),2,2);
      imshow("Hand gesture recognition", img );
      t.stop();
      std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms" << std::endl;
      char ch=waitKey(1);
      if(ch=='t')
      { break; }
    } 

    Ptr<Tracker> tracker;

    //tracker = TrackerBoosting::create();
    tracker = TrackerMIL::create();
    //tracker = TrackerKCF::create();
    
    // Read video
    VideoCapture video=cap;
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
      cout << "Could not read video file" << endl;
      return 1;  
    }
     
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
     
    // Define initial boundibg box
    
     
    // Uncomment the line below to select a different bounding box
    //bbox = selectROI(frame, false);
   
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Hand gesture recognition", frame);
     
    tracker->init(frame, bbox);
     
    while(video.read(frame))
    {
      cv::TickMeter t;
      t.start();
   
      // Update the tracking result
      bool ok = tracker->update(frame, bbox);
      
      xx1=int(bbox.x);
      yy1=int(bbox.y)-1.3*int(bbox.height);
      xx2=int(bbox.x)+int(bbox.width);
      yy2=int(bbox.y)+0.7*int(bbox.height);  

      if (ok)
      { // Tracking success : Draw the tracked object
        rectangle(frame, Point(xx1,yy1), Point(xx2,yy2), Scalar( 255, 0, 0 ), 2, 1 );
      }
      else
      { // Tracking failure detected.
        putText(frame, "Tracking failure detected", Point(70,35), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,0,0),2);
      }
      putText(frame, "Hand tracking step", Point(30,35), FONT_HERSHEY_SIMPLEX , 0.9, Scalar(255,0,0),2,2);
      
      string s;
      stringstream out;
      out<<classId;
      s=out.str();
      putText(frame, "Predicted class: "+s, Point(30,75), FONT_HERSHEY_SIMPLEX , 0.9, Scalar(55,47,114),2,2);

      // Display frame.
      imshow("Hand gesture recognition", frame);
      
      ///////extract image from frame
      Mat hand = frame(Rect(xx1,yy1,xx2-xx1,yy2-yy1));
   
      //! [Transformations]
      Mat bgr[3];
      split(hand,bgr);
      Mat thresh_r;
      Mat thresh_g;
      Mat thresh_b;
      threshold(bgr[2],thresh_r,140,255,THRESH_BINARY_INV);
      threshold(bgr[1],thresh_g,140,255,THRESH_BINARY_INV);
      threshold(bgr[0],thresh_b,140,255,THRESH_BINARY_INV);
      
      imshow("Hand", thresh_b);
       
      Mat hand_binary; 
      resize(thresh_b,hand_binary,Size(),0.23,0.23,INTER_CUBIC);
      //! [Transformations]
      
      Mat imge = hand_binary;

      //Model accepts only 48x48 one-channel images
      Mat inputBlob = blobFromImage(imge, 1.0f, Size(48, 48),
                  Scalar(), false);   //Convert Mat to batch of images


     /////// classification
      Mat prob;
      
      for (int i = 0; i < 1; i++)
      {
        //CV_TRACE_REGION("forward");
        //! [Set input blob]
        net.setInput(inputBlob);    //set the network input
        //! [Set input blob]
        //! [Make forward pass]
        prob = net.forward("prob");              //compute output
        //! [Make forward pass]
      }

      double classProb;
      getMaxClass(prob, &classId, &classProb);//find the best class
      t.stop();
      std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms" << std::endl;

      // Exit if ESC pressed.
      char k = waitKey(1);
      if(k == 't')
      { break; }
      
    }

    char kk = waitKey(1);
    if(kk == 'q')
    { break; }
    
  }
  
}


