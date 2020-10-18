#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
using namespace std;
using namespace cv;
#define debug true
int main() {
    std::cout << "Hello, World!" << std::endl;
//    Mat imageL = imread("./1.jpg");
//    Mat imageR = imread("./2.jpg");
//    String name1 = "./P01016-162951/up-1";
//    String name2 = "./P01016-162951/up-2";
    String name1 = "./P01016-162951/un-1";
    String name2 = "./P01016-162951/un-2";
    Mat imageL = imread(name1+".jpg");
//    Mat imageL = imread("./P01016-162951/up-1.jpg");
    Mat imageR = imread(name2+".jpg");
    if(imageL.empty()||imageR.empty()){
        cout<<"image is empty!"<<endl;
        return 1;
    }
    //可加入图片的反光处理
  //sift,surf algorithm
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.09;
    double edgeThreshold = 10;
    double sigma = 1.6;
    Ptr<SIFT> sift = SIFT::create(nfeatures, nOctaveLayers,
            contrastThreshold, edgeThreshold,
            sigma);
//    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create();
    vector<KeyPoint> keyPointL, keyPointR;
    sift->detect(imageL, keyPointL);
    sift->detect(imageR, keyPointR);
    //画特征点
    cv::Mat keyPointImageL;
    cv::Mat keyPointImageR;
    drawKeypoints(imageL, keyPointL, keyPointImageL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(imageR, keyPointR, keyPointImageR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //显示窗口
    cv::namedWindow("kpL",0);
    cv::namedWindow("kpR",0);

    resizeWindow("kpL", 0.5*imageL.cols, 0.5*imageL.rows);
    resizeWindow("kpR", 0.5*imageR.cols, 0.5*imageR.rows);

    //显示特征点
    cv::imshow("kpL", keyPointImageL);
    cv::imshow("kpR", keyPointImageR);
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    String filename = name1+"_"+to_string(nfeatures)+"_"+to_string(nOctaveLayers)+"_"+to_string(contrastThreshold)+"_"+to_string(edgeThreshold)+"_"+to_string(sigma);
    imwrite(filename+".png", keyPointImageL, compression_params);

    //特征点匹配
    cv::Mat despL, despR;
    //提取特征点并计算特征描述子
    sift->detectAndCompute(imageL, cv::Mat(), keyPointL, despL);
    sift->detectAndCompute(imageR, cv::Mat(), keyPointR, despR);
    cv::waitKey(0);

//    surf->detectAndCompute(imageL, cv::Mat(), keyPointL, despL);
//    surf->detectAndCompute(imageR, cv::Mat(), keyPointR, despR);

    //Struct for DMatch: query descriptor index, train descriptor index, train image index and distance between descriptors.
    //int queryIdx –>是测试图像的特征点描述符（descriptor）的下标，同时也是描述符对应特征点（keypoint)的下标。
    //int trainIdx –> 是样本图像的特征点描述符的下标，同样也是相应的特征点的下标。
    //int imgIdx –>当样本是多张图像的话有用。
    //float distance –>代表这一对匹配的特征点描述符（本质是向量）的欧氏距离，数值越小也就说明两个特征点越相像。
    std::vector<cv::DMatch> matches;

    //如果采用flannBased方法 那么 desp通过orb的到的类型不同需要先转换类型
    if (despL.type() != CV_32F || despR.type() != CV_32F)
    {
        despL.convertTo(despL, CV_32F);
        despR.convertTo(despR, CV_32F);
    }

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    matcher->match(despL, despR, matches);  //对despL中的每个点queryIdx，寻找despR中对应的匹配点trainIdx

    if(0) {
        cout<<"keypointsL.size = "<<keyPointL.size()<<"\tkeypointsR.size = "<<keyPointR.size()<<endl;
        cout<<"--queryIdx:"<<endl;
        for(int i = 0; i<matches.size(); i++)   cout<<matches[i].queryIdx<<",";
        cout<<endl<<"--trainIdx:"<<endl;
        for(int i = 0; i<matches.size(); i++)   cout<<matches[i].trainIdx<<",";
        cout<<endl;
    }
    //
    vector<Point2f> p2fi1, p2fi2;
    for(int i = 0;i<matches.size(); i++){
        p2fi1.push_back(Point2f(keyPointL[matches[i].queryIdx].pt));
        p2fi2.push_back(Point2f(keyPointR[matches[i].trainIdx].pt));
    }
    //用findFundamental来统计内点和外点以消除错误匹配
    vector<uchar> m_RANSACStatus;
    Mat F = findFundamentalMat(p2fi1, p2fi2, m_RANSACStatus, FM_RANSAC);
    vector<KeyPoint> kp_Ffilter1, kp_Ffilter2;
    vector<DMatch> matches_Ffilter;
    for(int i = 0; i<m_RANSACStatus.size(); i++)
    {
        if(m_RANSACStatus[i]!=0){
            //收集内点
            kp_Ffilter1.push_back(keyPointL[matches[i].queryIdx]);
            KeyPoint kp2 = keyPointR[matches[i].trainIdx];
//            kp_Ffilter1.push_back(kp1);
            kp_Ffilter2.push_back(kp2);
            //收集匹配
            matches_Ffilter.push_back(matches[i]);
        }
    }
    if(debug) {
        cout<<"filterkpl.size = "<<kp_Ffilter1.size()<<"\tkfilterkpr.size = "<<kp_Ffilter2.size()<<endl;

        cout<<"--queryIdx:"<<endl;
        for(int i = 0; i<matches_Ffilter.size(); i++)   cout<<matches_Ffilter[i].queryIdx<<",";

        cout<<endl<<"--trainIdx:"<<endl;
        for(int i = 0; i<matches_Ffilter.size(); i++)   cout<<matches_Ffilter[i].trainIdx<<",";

        cout<<endl<<"--querykp:"<<endl;
        for(int i = 0; i<matches_Ffilter.size(); i++)   cout<<kp_Ffilter1[i].pt<<",";

        cout<<endl<<"--trainpt:"<<endl;
        for(int i = 0; i<matches_Ffilter.size(); i++)   cout<<kp_Ffilter2[i].pt<<",";

        cout<<endl;
    }
    cout<<"filter1size"<<kp_Ffilter1.size()<<"\nfilter2size"<<kp_Ffilter2.size()<<"\nmatchFiltersize"<<matches_Ffilter.size()<<endl;
    // 用findHomography来找变换
    vector<uchar> h_RANSACStatus;
//    Mat H = findHomography(p2fi1, p2fi2,RANSAC);
//    Mat H = findHomography(p2fi1, p2fi2, h_RANSACStatus);   //看是否能用来消除匹配
//    perspectiveTransform()

    //计算特征点距离的最大值
    double maxDist = 0;
    double minDist = 99999;
    for (int i = 0; i < despL.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
            maxDist = dist;
        if(dist < minDist)
            minDist = dist;
    }
    cout<<"--Max dist: "<<maxDist<<"\n--Min dist: "<<minDist<<endl;

    //挑选好的匹配点
    std::vector< cv::DMatch > good_matches;
    for (int i = 0; i < despL.rows; i++)
    {
        if (matches[i].distance < max(minDist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }



    cv::Mat imageOutput;
//    cv::drawMatches(imageL, keyPointL, imageR, keyPointR, good_matches, imageOutput);
    cv::drawMatches(imageL, keyPointL, imageR, keyPointR, good_matches, imageOutput);

    cv::namedWindow("match",0);
    resizeWindow("match", 0.5*imageOutput.cols, 0.5*imageOutput.rows);
    cv::imshow("match", imageOutput);
    cv::waitKey(0);

    Mat image_FfilterOutput;
    drawMatches(imageL, kp_Ffilter1, imageR, kp_Ffilter2, matches_Ffilter, image_FfilterOutput);
    namedWindow("matches_Ffilter", 0);
    resizeWindow("matches_Ffilter", 0.5*image_FfilterOutput.cols, 0.5*image_FfilterOutput.rows);
    imshow("matches_Ffilter", image_FfilterOutput);
    waitKey(0);



    return 0;
}
