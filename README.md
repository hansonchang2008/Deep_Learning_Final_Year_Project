# Deep_Learning_Final_Year_Project
Essential Python Code for 1st Place Best Final Year Project:
* https://www.cis.um.edu.mo/bestfyp.html

Poster: https://www.cis.um.edu.mo/images/fyp_posters/20182019/2018_2019_01.jpg

Title:
Non-invasive Multi-disease Detection via Face and Tongue Image Analysis using Deep Learning

标题：基于深度学习的面部和舌象分析的无创多病种检测


The Code is organized as follows:
* 3 Classifiers:
  * CNN.py - The TensorFlow code for Convolutional Neural Network implementation (Deep Learning)
  * kNN.py - The k-Nearest Neighbor classifier implemented using Sci-kit Learn in Python
  * svm.py - The Support Vector Machine classifier implemented using Sci-kit Learn in Python

* Graphical User Interface:
  * Backend.py - The backend code for Graphical User Interface that uses the trained CNN model and predicts the health status
  * main_gui.py - The main code for Graphical User Interface using Pyqt5 Package in Python
  * my_gui_change_backgroud.py - The code to change the background of the GUI

* Data Preprocessing:
  * extraction_facial.py - The code extracts four facial blocks from the face image, achieved by locating the positions of eyes first
  * extraction_tongue.py - The code extracts eight tongue blocks from the tongue image
  * combine_blocks.py - The code combines the twelve image blocks into one

Abstract:
Heart disease and diabetes are two major causes of death all over the world. While the traditional detection methods for both diseases are invasive or inconvenient. To simplify the detection of both illnesses, non-invasive detection approaches are needed. In this project, we developed an Intelligent Disease Detection (iDD) system, a non-invasive multi-class disease detection system based on face and tongue image analysis. A user-friendly graphical user interface (GUI) is developed to make our system for public use. Our iDD system can determine a person’s health status in real-time.

摘要：心脏病和糖尿病是全世界最致死的两种疾病。然而，他们的传统检测方法是侵入性的或者不方便的。为了简化这两个疾病的检测，开发非侵入性检测方式是有必要的。这个项目中我们开发了智能疾病检测系统（iDD），一个基于脸和舌头分析的非侵入性的多类的疾病检测系统。也开发了一个用户友好的图形界面程序让我们的系统可以供公众使用。我们的智能疾病监测系统可以实时检测一个人的健康状况。

Installation requirements: OS Ubuntu 18.04, Disk Space 1GB

The Whole face and tongue images are captured using special devices to reduce noise.
