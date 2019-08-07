
CNN-based hand gesture interface
================================

Hand gesture interface for Desktop PC and Raspberry Pi. The system is intended to recognize 10 hand gestures. Based on such poses you can control some devices such as a drone, mobile robots, screens among others.

The systems work in real-time. The Convolutional Neural Networks (CNNs) were trained using Caffe framework. OpenCV libraries were used. In order to obtain a better computation performance, the system was implemented using C++ language. The systems are intended to work on Linux systems, however, can be compiled for Windows or other Operating System.

In order to modify the code, you should compile with:

```
$ make
```

Finally, execute the created file. Example:

```
$ ./handgesture_detrack
```

## License ##

GNU General Public License, version 3 (GPLv3).

You can visit my personal website: [http://dennishnf.bitbucket.io](http://dennishnf.bitbucket.io)!.


