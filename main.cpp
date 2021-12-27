#include <iostream>
#include "opencv2/opencv.hpp"
#include <chrono>
using namespace std;
using namespace cv;





cv::Mat applyConv(cv::Mat & oldImage,cv::Mat_<float> & kernel){


    Mat newImage(oldImage.rows,oldImage.cols,oldImage.type()); // image to be returned
    Mat Img_Mat(oldImage);
    std::vector<Mat> BGR; //vector contains each channel separatley
    std::vector<Mat> newBGR;
    split(Img_Mat, BGR);
    split(Img_Mat, newBGR);

    Mat_<float> flipped_kernel;
    flip(kernel, flipped_kernel, -1);

    const int dx = kernel.cols / 2;
    const int dy = kernel.rows / 2;

    for (long i = 0; i<oldImage.rows; i++)
    {
        for (long j = 0; j<oldImage.cols; j++)
        {
            float tmp = 0;
            float tmp2 = 0;
            float tmp3= 0;

            for (long k = 0; k<flipped_kernel.rows; k++)
            {
                for (long l = 0; l<flipped_kernel.cols; l++)
                {

                    int x = j - dx + l;
                    int y = i - dy + k;

                    if (x >= 0 && x < oldImage.cols && y >= 0 && y < oldImage.rows)
                        //tmp += oldImage.at<float>(y, x) * flipped_kernel.at<float>(k, l);
                        tmp+=(float)BGR[0].at<uchar>(y,x)* flipped_kernel.at<float>(k, l);
                        tmp2+=(float)BGR[1].at<uchar>(y,x)* flipped_kernel.at<float>(k, l);
                        tmp3+=(float)BGR[2].at<uchar>(y,x)* flipped_kernel.at<float>(k, l);
                }

            }



            newBGR[0].at<uchar>(i,j)=saturate_cast<uchar>(tmp);
            newBGR[1].at<uchar>(i,j)=saturate_cast<uchar>(tmp2);
            newBGR[2].at<uchar>(i,j)=saturate_cast<uchar>(tmp3);
            }



        }

    merge(newBGR,newImage);
    return newImage;
    }









int main() {


    //kernels definition
    Mat_<float> edgeDetection(3, 3);
    Mat_<float> blur(3, 3);
    Mat_<float> sharpening(3,3);
    Mat_<float> identity(3,3);
    Mat_<float> emboss(3,3);



    edgeDetection << 0, -1, 0, -1, 4, -1, 0, -1, 0; //edge detection edgeDetection
    sharpening<< 0, -1, 0, -1, 5, -1, 0, -1, 0;
    identity << 0, 0, 0, 0, 1, 0, 0, 0 ,0;
    emboss << -2, -1, 0, -1, 1, 1,0, 1, 2;
    blur<< 0.1111, 0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1111;
    //////////////////kernels definition finshed///////////////////

    Mat image = imread("/Users/josephdaoud/downloads/pexels-maxime-francis-2246476.jpg");
    Mat sideBySide;

    if (image.empty()) {
        cout << "Image File "
             << "Not Found" << endl;


        cin.get();
        return -1;
    }



    auto startime=std::chrono::high_resolution_clock::now();
    cv::Mat filteredImage=applyConv(image, edgeDetection);
    auto endtime= chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(endtime - startime);

    cout<<"Total execution time in ms : "<<duration.count();
    hconcat(image,filteredImage,sideBySide);
    imshow("iFilter", sideBySide);





    waitKey(0);

    return 0;
}
