#include <opencv2/opencv.hpp>

#include "mv_camera.hpp"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <math.h>
#include <vector>
#include <iostream>
#include <fmt/color.h>


const bool debug_ = true;
int debug_frame_num = 0;


void camera_test_debug(){
    Devices::MV_Camera c;
    c.open();
    cv::Mat read_img;
    cv::Mat img;
    namedWindow("test",cv::WINDOW_NORMAL);

    while(true)
    {
        c.read(read_img);
        img = read_img.clone();
        
        cv::imshow("test",img);
        
        if(cv::waitKey(1) == 27)
        {
            break;
        }
    }
    c.close();
}

class box
{
    public:
        cv::Rect box;
        double iou;
};


double min(double a,double b)
{
    if (a > b)
        return b;
    else
        return a;
}

double max(double a,double b)
{
    if (a > b)
        return a;
    else
        return b;
}

double dist(double x1,double y1,double x2,double y2)
{
    double result = pow((pow((x1-x2),2)) + pow((y1-y2),2),0.5);
    return result;
}

double IOU(cv::Rect rect1,cv::Rect rect2)
{
    double result;
    double xa,ya,xb,yb;
    double W,H,jiao,bing;

    xa = max(rect1.x,rect2.x);
    ya = max(rect1.y,rect2.y);
    xb = min(rect1.x + rect1.width,rect2.x + rect2.width);
    yb = min(rect1.y + rect1.height,rect2.y + rect2.height);


    W = abs(xa - xb);
    H = abs(ya - yb);

    if(xb <= xa || yb <= ya)
    {
        W = 0;
        H = 0;
    }

    jiao = W * H;
    
    bing = rect1.width * rect1.height + rect2.width * rect2.height - jiao;
    
    result = jiao / bing;

    return result;
}

double GIOU(cv::Rect rect1,cv::Rect rect2)
{

    double xa,ya,xb,yb;
    double W,H,jiao,bing;

    double xc,yc,xd,yd;
    double CW,CH,C,C_;
    double result;


    xa = max(rect1.x,rect2.x);
    ya = max(rect1.y,rect2.y);
    xb = min(rect1.x + rect1.width,rect2.x + rect2.width);
    yb = min(rect1.y + rect1.height,rect2.y + rect2.height);


    W = abs(xa - xb);
    H = abs(ya - yb);

    jiao = W * H;
    
    bing = rect1.width * rect1.height + rect2.width * rect2.height - jiao;

    xd = min(rect1.x,rect2.x);
    yd = min(rect1.y,rect2.y);
    xc = max(rect1.x + rect1.width,rect2.x + rect2.width);
    yc = max(rect1.y + rect1.height,rect2.y + rect2.height);

    CW = abs(xd - xc);
    CH = abs(yd - yc);

    C = CW * CH;

    C_ = C - bing;

    result = IOU(rect1,rect2) - C_/C;

    return result;


}

double DIOU(cv::Rect rect1,cv::Rect rect2)
{
    double xc,yc,xd,yd;
    double c,d;

    double xe,ye,xf,yf;

    double result;

    xd = min(rect1.x,rect2.x);
    yd = min(rect1.y,rect2.y);
    xc = max(rect1.x + rect1.width,rect2.x + rect2.width);
    yc = max(rect1.y + rect1.height,rect2.y + rect2.height);

    c = dist(xd,yd,xc,yc);

    xe = min(rect1.x + rect1.width/2 , rect2.x + rect2.width/2);
    ye = min(rect1.y + rect1.height/2 , rect2.y + rect2.height/2);
    xf = max(rect1.x + rect1.width/2 , rect2.x + rect2.width/2);
    yf = max(rect1.y + rect1.height/2 , rect2.y + rect2.height/2);

    d = dist(xe,ye,xf,yf);

    result = (d*d)/(c*c);

    return result;

}

double DIOU_loss(cv::Rect rect1,cv::Rect rect2)
{
    double result;
    result = 1 - IOU(rect1,rect2) + DIOU(rect1,rect2);
    return result;
}


int main() {

    //开相机测试
    #if 0
        if(debug_) camera_test_debug();
    #endif
    
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT,cv::Size(10,10));
    int hmin1 = 2, smin1 = 39, vmin1 = 255;
    int hmax1 = 39, smax1 = 255, vmax1 = 255;

    std::vector<std::vector<cv::Point>> contours;   //放二值化后检测到的轮廓用的向量
    std::vector<cv::Vec4i> hierarchy;   //cv::findContours中必须用的向量，该卡尔曼滤波未使用此功能，照着这样写就行
    cv::Point2f pts[4];    
    
    double last_width[100] = {0};
    double last_height[100] = {0};
    double last_rect_x[100] = {0};
    double last_rect_y[100] = {0};
    double last_area[100] = {0};
    double rect_x, rect_y, area;
    double ratio = 0.1;
    std::vector<box> R_box(100);
    std::vector<box> last_R_box;
    box r_box;



    // double iou;
    // std::vector<double> iou_vector;

    // int frame_m = 0;

    // cv::VideoCapture capture("/home/fuziming/MA/buffkaigan/1.0/test.MP4");  
    cv::VideoCapture capture("/home/fuziming/MA/buffkaigan/1.0/test.MP4");  
  
    if (!capture.isOpened())  
    {  
        std::cout << "Read video Failed !" << std::endl;  
    }   
  
    cv::Mat frame;  
    cv::namedWindow("video test");  
  
    int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);  
    // fmt::print(fmt::bg(fmt::color::blue),"Current frame num is {}\n", frame_num);
    
    bool is_first_frame = true;
    cv::Mat previous_frame;

    int n = 0;

    for (int i = 0; i < frame_num - 1; ++i)  
    {  


        try {
            // 检查摄像头是否成功打开
            if (!capture.isOpened()) {
                throw std::runtime_error("无法打开视频文件");
            }
            
            // 读取视频帧
            capture.read(frame);
        }
        catch (const std::exception& e) {
            std::cerr << "发生异常: " << e.what() << std::endl;
        }

        if(debug_){
            try {

                // 检查图片是否成功读取
                if (frame.empty()) {
                    throw std::runtime_error("无法读取图片");
                }

                
                fmt::print(fmt::fg(fmt::color::red),"Current frame is {}\n", n);
                n += 1;

                }
            catch (const std::exception& e) {
                fmt::print(fmt::fg(fmt::color::green),"发生异常:  {}\n", e.what());
            }
        }

        

  
    #if 1
        cv::Mat img_hsv;
        cv::Mat mask;
        cv::Mat kai;
        //test

        cv::cvtColor(frame, img_hsv, cv::COLOR_BGR2HSV);


        cv::Scalar lower1(hmin1, smin1, vmin1);
		cv::Scalar upper1(hmax1, smax1, vmax1);
		cv::inRange(img_hsv, lower1, upper1, mask);
        cv::morphologyEx(mask,kai,cv::MORPH_DILATE,kernel,cv::Point(-1,-1),1);
        cv::findContours(kai, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if(contours.size() > 0)
        {
        for (int i = 0;i < contours.size(); ++i)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            cv::rectangle(frame,rect,cv::Scalar(0,255,0),3);


            // std::cout<<rect<<std::endl;
            // 可以用 rect.x, rect.y, rect.width, rect.height 获取方框数据
            
            r_box.box = rect;

            R_box.push_back(r_box);


            
        }

        


        for(int i = 0; i <= 0; i++)
        {
            
            if(last_R_box.empty()){
                break;
            }
            
            for(int j = 0; j <= R_box.size() ; j++)
            {             
             
                R_box[j].iou = IOU(last_R_box[i].box,R_box[j].box);

            }
        }
     

        // std::cout << R_box[0].iou << std::endl; 
        // std::cout << R_box[1].iou << std::endl; 
        // std::cout << R_box[2].iou << std::endl; 
        // std::cout << R_box[3].iou << std::endl; 
        // std::cout << R_box[4].iou << std::endl; 
        // std::cout << R_box[5].iou << std::endl; 


        std::sort(R_box.begin(),R_box.end(), [](const box & a,const box & b){return a.iou > b.iou;});
        
        
       

        std::cout << R_box[0].iou << std::endl; 
        std::cout << R_box[1].iou << std::endl; 
        std::cout << R_box[2].iou << std::endl; 
        std::cout << R_box[3].iou << std::endl; 
        std::cout << R_box[4].iou << std::endl; 
        std::cout << R_box[5].iou << std::endl; 



        if(!R_box.empty())
        {
            cv::rectangle(frame,R_box[0].box,cv::Scalar(0,0,255),3);
        }
        last_R_box = R_box;




        // //test
        fmt::print(fg(fmt::color::orange),"R_box_vector_size is : {}\n",R_box.size());
        R_box.clear();

        }

        cv::imshow("video test", frame);  

        // cv::imshow("HSV",img_hsv);
        // cv::imshow("Mask",mask);
        cv::imshow("kai",kai);



        // Storing img for debugging
        #if 1
        if(debug_){
            std::ostringstream oss;
            std::string file_name = "/home/fuziming/MA/buffkaigan/1.0/debug_img/";

            oss << file_name << debug_frame_num << ".jpg";
            cv::imwrite(oss.str(), frame);

            debug_frame_num++;
        }
        #endif




        if (cv::waitKey(1) == 'q')  
        {  
            break;  
        }  
        #endif
     
    }  
  
    cv::destroyWindow("video test");  
    capture.release();  
    return 0;  


}