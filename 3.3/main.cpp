#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "mv_camera.hpp"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <math.h>
#include <cmath>

#include <vector>
#include <iostream>
#include <fmt/color.h>

#define PI 3.141592653579

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
        std::vector<cv::Point> contours;   //放二值化后检测到的轮廓用的向量
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
}//iou以外的iou变体函数有问题，后面如有需要要重写；

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


//将向量里的元素降序排序(box形式)
void sort_des_box(std::vector<box>& a_box)
{
    std::sort(a_box.begin(),a_box.end(), [](const box & a,const box & b){return a.iou > b.iou;});
}


//将向量里的元素降序排序(double形式)
void sort_des_double(std::vector<double>& a_box)
{
    std::sort(a_box.begin(),a_box.end(), [](const double & a,const double & b){return a > b;});
}

//输入四个点的数组，自动求四个点组成的矩形的中点
cv::Point2f rotatedrect_center(cv::Point2f pts[4])
{
    int x,y;
    cv::Point2f center;
    x = (pts[0].x + pts[2].x) / 2;
    y = (pts[0].y + pts[2].y) / 2;
    center.x = x;
    center.y = y;
    return center;
}

//计算cv::Rect box的中心点(box类版)
cv::Point2f rect_center(box r_box)
{
    cv::Point2f center;
    double x,y,w,h;
    x = r_box.box.x;
    y = r_box.box.y;
    w = r_box.box.width;
    h = r_box.box.height;

    center.x = x + w / 2;
    center.y = y + h / 2;

    return center;
}

//计算cv::Rect box的中心点
cv::Point2f rect_center1(cv::Rect r_box)
{
    cv::Point2f center;
    double x,y,w,h;
    x = r_box.x;
    y = r_box.y;
    w = r_box.width;
    h = r_box.height;

    center.x = x + w / 2;
    center.y = y + h / 2;

    return center;
}

//为下面的函数计算角度
cv::Point2f angle_degrees_compute(cv::Point2f c, cv::Point2f p, double r, double angle_degrees)
{
    cv::Point2f new_point;
    double theta = atan2(p.y - c.y,p.x - c.x);

    //角度转弧度
    double angle_radians = angle_degrees * PI / 180; 

    double new_theta = theta + angle_radians;
    new_point.x = c.x + r * cos(new_theta);
    new_point.y = c.y + r * sin(new_theta);
    return new_point;


}

//根据第一个扇叶的位置计算出其他扇叶的位置中心点
std::vector<cv::Point2f> first_estimate(cv::Point2f c,cv::Point2f p,cv::Mat frame)
{
    double angle_degrees = 72;
    double r = dist(c.x,c.y,p.x,p.y);
    cv::Point2f another_point;

    std::vector<cv::Point2f> all_point;

    for(int i = 0; i < 360; i = i+72)
    {
        another_point = angle_degrees_compute(p,c,r,i);
        all_point.push_back(another_point);
    }
    fmt::print(fg(fmt::color::blue),"all_point.size is : {}\n",all_point.size());
    for(int i = 0; i < 5; i++)
    {
        cv::circle(frame,all_point[i],5,cv::Scalar(0,255,0),-1);
    }

    return all_point;
    
}


//已知矩形的中心点，画一个cv::Rect矩形，根据另一个矩形的长和宽
cv::Rect rect_by_center(cv::Point2f center,cv::Rect rect)
{
    double width = rect.width;
    double height = rect.height;

    cv::Rect rectangle(center.x - width/2, center.y - height/2, width, height);

    return rectangle;
}

//根据计算出的中心点，画出所有估算的矩形
void draw_all_rect(std::vector<cv::Point2f> first_e,cv::Rect rect,cv::Mat frame)
{
    std::vector<cv::Rect> all_fer;
    cv::Rect fe_rect;
    fmt::print(fg(fmt::color::green),"first_e.size is : {}\n",first_e.size());
    for(int i = 0; i < first_e.size(); i++)
    {
        fe_rect = rect_by_center(first_e[i],rect);
        all_fer.push_back(fe_rect);
    }
    fmt::print(fg(fmt::color::pink),"all_fer.size is : {}\n",all_fer.size());

    for(int i = 0; i < all_fer.size(); i++)
    {
        cv::rectangle(frame,all_fer[i],cv::Scalar(0,255,0),2);
    }
}




//该函数用于处理被两圆遮挡后的二值化图像
void process_mask1(cv::Mat mask1,cv::Point2f R_center,cv::Mat frame)
{
    std::vector<std::vector<cv::Point>> contours;   //放二值化后检测到的轮廓用的向量
    std::vector<cv::Vec4i> hierarchy;   //cv::findContours中必须用的向量,不需要赋值
    cv::Point2f pts[4];    
    cv::Point2f first_leaf_center; 
    std::vector<cv::Point2f> leaf_center_pts(100);

    cv::Mat rotate_mask1;
    cv::Mat rotate_mask2;
    cv::Mat rotationMatrix;//旋转矩形
    double angle = 72;  // 旋转角度
    rotationMatrix = cv::getRotationMatrix2D(R_center, angle, 1.0);

    std::vector<cv::Point2f> first_e; //该向量储存通过第一个扇叶估算出的所有扇叶中心点的坐标


    



    cv::findContours(mask1, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if(contours.size() > 0)
    {
        for(int i = 0; i < contours.size() ; i++)
        {
            
            cv::RotatedRect rect1 = cv::minAreaRect(contours[i]);   //创建最小的能够框出检测到的轮廓的旋转矩形
            cv::Rect rect = cv::boundingRect(contours[i]);
            rect1.points(pts);

            // for(int j = 0;j < 4;j++)
            // {
	        //     line(frame, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 255, 255), 3, 8);  //绘制最小外接矩形每条边
            // }

            // cv::rectangle(frame,rect,cv::Scalar(255,255,255),3);

            if(contours.size() == 1)
            {
                // first_leaf_center = rotatedrect_center(pts);//计算第一个扇叶中心点坐标
                leaf_center_pts.push_back(first_leaf_center);
                cv::rectangle(frame,rect,cv::Scalar(255,255,255),3);

                first_e = first_estimate(rect_center1(rect),R_center,frame);//通过第一个扇叶估算出的所有扇叶中心点的坐标
                draw_all_rect(first_e,rect,frame);

            }
            
            






            //当新出现轮廓时开始判断，若新轮廓在iou区域内，确定为新识别到的扇叶

        }
    }
    fmt::print(fg(fmt::color::orange),"contours is : {}\n",contours.size());
    

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

    cv::Point2f R_center;
    cv::Point2f first_fan_leaf_pts[4];
    std::vector<double> pts_to_R(4);
    double R_distance = 0;
    // cv::Point2f RR_center;//test


    double iou;
    std::vector<double> iou_vector;

    int frame_m = 0;

    // cv::VideoCapture capture("/home/fuziming/MA/buffkaigan/1.0/test.MP4");  
    #if 1
    cv::VideoCapture capture("/home/fuziming/MA/buffkaigan/3.1/test4.MP4");  
  
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

        // if(debug_){
        //     try {

        //         // 检查图片是否成功读取
        //         if (frame.empty()) {
        //             throw std::runtime_error("无法读取图片");
        //         }

                
        //         fmt::print(fmt::fg(fmt::color::red),"Current frame is {}\n", n);
        //         n += 1;
        // std::sort(R_box.begin(),R_box.end(), [](const box & a,const box & b){return a.iou > b.iou;});

        //         }
        //     catch (const std::exception& e) {
        //         fmt::print(fmt::fg(fmt::color::green),"发生异常:  {}\n", e.what());
        //     }
        // }
    #endif 
        

  
    #if 1
        cv::Mat img_hsv;
        cv::Mat mask;
        cv::Mat kai;
        cv::Mat mask1;
        //test

        cv::cvtColor(frame, img_hsv, cv::COLOR_BGR2HSV);


        cv::Scalar lower1(hmin1, smin1, vmin1);
		cv::Scalar upper1(hmax1, smax1, vmax1);
		cv::inRange(img_hsv, lower1, upper1, mask);
        cv::morphologyEx(mask,kai,cv::MORPH_DILATE,kernel,cv::Point(-1,-1),1);
        cv::findContours(kai, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        mask1 = kai.clone();

        if(contours.size() > 0)
        {
        for (int i = 0;i < contours.size(); ++i)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            cv::RotatedRect rect1 = cv::minAreaRect(contours[i]);   //创建最小的能够框出检测到的轮廓的旋转矩形
            rect1.points(pts);
            for (int j = 0; j < 4; j++)
            {
	            line(frame, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 3, 8);  //绘制最小外接矩形每条边
            }
            // cv::rectangle(frame,rect,cv::Scalar(0,255,0),3);


            // std::cout<<rect<<std::endl;
            // 可以用 rect.x, rect.y, rect.width, rect.height 获取方框数据
            
            r_box.box = rect;

            r_box.contours = contours[i];

     

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
        
        sort_des_box(R_box);



        if(!R_box.empty())
        {
            cv::rectangle(frame,R_box[0].box,cv::Scalar(0,0,255),3);
        }

        if(R_box.size() == 2)
        {
            cv::rectangle(frame,R_box[1].box,cv::Scalar(255,0,0),3);
            cv::RotatedRect first_fan_leaf = cv::minAreaRect(R_box[1].contours);   //创建最小的能够框出检测到的轮廓的旋转矩形
            first_fan_leaf.points(first_fan_leaf_pts);
            for (int j = 0; j < 4; j++)
            {
	            line(frame, first_fan_leaf_pts[j], first_fan_leaf_pts[(j + 1) % 4], cv::Scalar(0, 100, 100), 3, 8);  //绘制最小外接矩形每条边
            }

            for(int i = 0;i < 4;i++)
            {
                pts_to_R[i] = dist(first_fan_leaf_pts[i].x,first_fan_leaf_pts[i].y,R_center.x,R_center.y);
            }

            sort_des_double(pts_to_R);
            R_distance = pts_to_R[0];
            // std::cout << R_distance << std::endl;

            // // test
            // RR_center = rect_center(first_fan_leaf_pts);

        }
        int F = abs(R_distance * 0.53 - 3);
        // cv::circle(frame , cv::Point2f(R_box[0].box.x + R_box[0].box.width / 2 , R_box[0].box.y + R_box[0].box.height / 2) , R_distance , cv::Scalar(255,255,255) , 3);
        // cv::circle(frame , cv::Point2f(R_box[0].box.x + R_box[0].box.width / 2 , R_box[0].box.y + R_box[0].box.height / 2) , R_distance * 0.53 , cv::Scalar(255,255,255) , 3);

        if(R_box.size() >= 2)
        {
        cv::circle(mask1 , cv::Point2f(R_box[0].box.x + R_box[0].box.width / 2 , R_box[0].box.y + R_box[0].box.height / 2) , R_distance , cv::Scalar(0,0,0) , 3);
        cv::circle(mask1 , cv::Point2f(R_box[0].box.x + R_box[0].box.width / 2 , R_box[0].box.y + R_box[0].box.height / 2) , R_distance * 0.53 , cv::Scalar(0,0,0) , -1);

        //接下来对mask1中的圆环中的轮廓进行处理
        process_mask1(mask1,R_center,frame);

        }
        // //test
        // cv::circle(frame , RR_center , R_distance , cv::Scalar(255,255,255) , 3);
        // cv::circle(frame , RR_center , R_distance * 0.53 , cv::Scalar(255,255,255) , 3);


        last_R_box = R_box;
        R_center.x = last_R_box[0].box.x + last_R_box[0].box.width / 2;
        R_center.y = last_R_box[0].box.y + last_R_box[0].box.height / 2;





        // //test
        // fmt::print(fg(fmt::color::orange),"R_box_vector_size is : {}\n",R_box.size());

        
        R_box.clear();


        







    

        }

        cv::putText(frame,"R",cv::Point(R_box[0].box.x,R_box[0].box.y - 4),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255),4);
        cv::imshow("video test", frame);  

        // cv::imshow("HSV",img_hsv);
        // cv::imshow("Mask",mask);

        cv::imshow("kai",kai);
        cv::imshow("mask1",mask1);



        // Storing img for debugging
        #if 1
        if(debug_){
            std::ostringstream oss;
            std::string file_name = "/home/fuziming/MA/buffkaigan/图片分析/frame/";

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

//中心位置不准确，总是偏上一些