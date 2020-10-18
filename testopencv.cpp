//
// Created by wrx on 2020/10/14.
//
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
void testimshow(){
    Mat src = imread("1.jpg");
    imshow("src", src);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY); // 注意，有的教程CV_BGR2GRAY，opencv4下会报错
    imshow("src_gray", src_gray);
    imwrite("eagle_gray.jpg", src_gray);
    waitKey(0);
}

//这个函数是测试do{} while(0)的功能
//答案是：执行do{}一次然后跳出
 void testdowhile(){
    do{
        int i = 0;
        cout<<i++<<endl;
    }
    while(0);
}

int main(int argc, char const *argv[])
{
//    testimshow();
    testdowhile();
    return 0;
}