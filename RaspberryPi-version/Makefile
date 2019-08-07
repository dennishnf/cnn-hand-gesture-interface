CC = g++
CFLAGS = -g -Wall
SRCS = handgesture_detrack.cpp
PROG = handgesture_detrack
     
OPENCV_LIBS = /usr/local
     
OPENCV = -I$(OPENCV_LIBS)/include/opencv -I$(OPENCV_LIBS)/include \
$(OPENCV_LIBS)/lib/libopencv_calib3d.so $(OPENCV_LIBS)/lib/libopencv_core.so \
$(OPENCV_LIBS)/lib/libopencv_features2d.so $(OPENCV_LIBS)/lib/libopencv_flann.so \
$(OPENCV_LIBS)/lib/libopencv_highgui.so $(OPENCV_LIBS)/lib/libopencv_imgproc.so \
$(OPENCV_LIBS)/lib/libopencv_ml.so $(OPENCV_LIBS)/lib/libopencv_objdetect.so \
$(OPENCV_LIBS)/lib/libopencv_photo.so $(OPENCV_LIBS)/lib/libopencv_stitching.so \
$(OPENCV_LIBS)/lib/libopencv_superres.so $(OPENCV_LIBS)/lib/libopencv_videoio.so \
$(OPENCV_LIBS)/lib/libopencv_videostab.so $(OPENCV_LIBS)/lib/libopencv_tracking.so \
$(OPENCV_LIBS)/lib/libraspicam.so $(OPENCV_LIBS)/lib/libraspicam_cv.so \
$(OPENCV_LIBS)/lib/libopencv_dnn.so \
$(OPENCV_LIBS)/lib/libopencv_imgcodecs.so \
     
LIBS = $(OPENCV)
     
$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
