
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>

#include<string>
#include<iostream>

//leaf_selection = 1 为检测待击打和已击打的全部扇叶为只检测待击打
//leaf_selection = 2 为只检测待击打
int leaf_selection = 2;

float cof_threshold = 0.8;
float nms_area_threshold = 0.5;


//Configure constant parameters
std::string IMAGE_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/5/0a1c6c12-3785.jpg";
std::string VIDOE_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/5/关灯-红方大能量机关-失败后激活成功的全激活过程.MP4";
// std::string VIDOE_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/5/关灯-蓝方大能量机关-全激活过程.MP4";
std::string CAMERA_PATH = "0";
std::string MODEL_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/5/model2/test2/weights/best_openvino_model/best.xml";
std::string DEVICE = "GPU";
cv::Mat img1;
// cv::Mat roi;
// img1 = cv::imread(IMAGE_PATH);
cv::VideoCapture capture;

float PI = 3.141592653579;
float g = 9.780;
float distance = 6.8;
float dh = 0.1;//单位像素对应的实际距离
float bullet_speed = 20;//弹速

cv::Point3f send_point;//最终发送坐标


// //弹道补偿角(传入值a1为现在枪口抬起角度，是弧度)
// float TC(float a1,float img_h,cv::Point2f tergrt_point,cv::Mat dst)
// {
//     float h = img_h * dh;
//     float H = distance * tan(a1);
//     float z0 = H - h;

//     float A1 = atan(z0/distance);
//     float l = sqrt(pow(distance,2) + pow(z0,2));
//     float A2 = asin((z0 + g * pow(distance,2) / bullet_speed)/l);
//     float A3 = (A2 + A1)/2;//A3为计算出的枪口应抬角度
    
//     float H1 = distance * A3;
//     float H2 = H1 - z0;

//     tergrt_point = cv::Point2f(tergrt_point.x,tergrt_point.y - h);

//     cv::circle(dst,tergrt_point,5,cv::Scalar(0,255,0),-1);

//     return A3;
// }




//用于求扇叶中心点
//                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14
//object_keypoints:x0,y0,s0,x1,y1,s1,x2,y2,s2,x3,y3,s3,x4,y4,s4
cv::Point2f leaf_center(cv::Mat img,std::vector<float> object_keypoints)
{
    int x,y;
    cv::Point2f center;
    x = (object_keypoints[0] + object_keypoints[3] + object_keypoints[9] + object_keypoints[12]) / 4;
    y = (object_keypoints[1] + object_keypoints[4] + object_keypoints[10] + object_keypoints[13]) / 4;
    center.x = x;
    center.y = y;
    cv::circle(img,center,5,cv::Scalar(0,255,0),-1);
    return center;
}



void letterbox(const cv::Mat& source, cv::Mat& result)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
}



int main() {

    cv::namedWindow("dst", cv::WINDOW_NORMAL);
    // cv::namedWindow("roi",cv::WINDOW_NORMAL);
    


    auto last_time = std::chrono::system_clock::now();
    auto new_time = std::chrono::system_clock::now();

    //扩展卡尔曼变量
    Eigen::Matrix<double,4,1> X;
    //      X  vx   y  vy
    X <<    0,  0,  0,  0;

    //上一次运算坐标位置
    Eigen::Matrix<double,2,1> X1;
    //     lx  ly  
    X1 <<   0,  0;

    //测量值矩阵
    Eigen::Matrix<double,2,1> X2;

    Eigen::Matrix<double,4,4> P;
    P.setIdentity();

    Eigen::Matrix<double,4,4> F;
    F.setIdentity();


    float t0 = 0.02;

    //加速度a(待测量估计)
    // double a = 0.000000000000002;
    // double a = 1;
    double a = 0.1;


    double variable1;
    double variable2;
    double variable3;

    Eigen::Matrix<double,4,4> Q;
    Q.setIdentity();

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> K;

    Eigen::Matrix<double,2,4> H;
    Eigen::Matrix<double,2,2> R;

    //单位矩阵
    Eigen::Matrix<double,4,4> I;
    I.setIdentity();

    cv::Mat X3(2,1,CV_64F);






	capture.open(VIDOE_PATH);
	if (!capture.isOpened()) {
		printf("could not load video data...\n");
		return -1;
	}

	int frames = capture.get(cv::CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
	double fps = capture.get(cv::CAP_PROP_FPS);//获取每针视频的频率

	// cv::namedWindow("video-demo", cv::WINDOW_AUTOSIZE);


	//1.Create Runtime Core
	ov::Core core;

	//2.Compile the model
	ov::CompiledModel compiled_model = core.compile_model(MODEL_PATH, DEVICE);

	//3.Create inference request
	ov::InferRequest infer_request = compiled_model.create_infer_request();

    // clock_t start_time = 0, end_time = 0;


	for(int i = 0; i < frames; i++)
	{
        // clock_t start_time, end_time;
		// start_time = clock();
        // end_time = clock();
		// std::cout << "Time:" << (double)(end_time - start_time) << "ms" << std::endl;

        // end_time = clock();
        // std::cout << "Time:" << (double)(end_time - start_time) << "ms" << std::endl;
		// start_time = clock();

        cv::Size windowSize = cv::getWindowImageRect("dst").size();

        // 计算窗口的中心点坐标
        int centerX = windowSize.width / 2;
        int centerY = windowSize.height / 2;

        cv::Point2f center(centerX,centerY);




        //时间戳
        auto end_time = std::chrono::system_clock::now();       //记下现在的时间

        new_time = end_time;
        // auto waste_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();
        // auto waste_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
        auto waste_time = std::chrono::duration_cast<std::chrono::microseconds>(new_time-last_time).count();
        last_time = end_time;       //更新时间
        double t = waste_time * 0.000001;   //微妙转换成秒  t为每次循环所用的时间
                

		capture >> img1;
		cv::Mat letterbox_img;
		letterbox(img1,letterbox_img);
		// cv::imshow("letterbox_img1",letterbox_img);
		float scale = letterbox_img.size[0] / 640.0;
		cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);


		auto input_port = compiled_model.input();

		ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));

		infer_request.set_input_tensor(input_tensor);

		// clock_t start_time, end_time;
		// start_time = clock();
		infer_request.infer();
		// end_time = clock();
		// std::cout << "Inference Time:" << (double)(end_time - start_time) << "ms" << std::endl;



		//Get output
		auto output = infer_request.get_output_tensor(0);
		auto output_shape = output.get_shape();


		float* data = output.data<float>();
		cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
		transpose(output_buffer, output_buffer); //[8400,23]
        cv::Mat dst = img1.clone();
        std::vector<int> class_ids;
        int class_id;
        

        
        for(int j = 4; j < 8; j = j + leaf_selection){

            std::vector<float> class_scores;
            std::vector<cv::Rect> boxes;
            std::vector<std::vector<float>> objects_keypoints;
            


            for (int i = 0; i < output_buffer.rows; i++) {
                // float class_score = output_buffer.at<float>(i, 4);
                float class_score = output_buffer.at<float>(i, j);

                if (class_score > cof_threshold) {
                    class_scores.push_back(class_score);
                    class_ids.push_back(j - 4); //{0:"person"}
                    class_id = j-4;
                    float cx = output_buffer.at<float>(i, 0);
                    float cy = output_buffer.at<float>(i, 1);
                    float w = output_buffer.at<float>(i, 2);
                    float h = output_buffer.at<float>(i, 3);
                    // Get the box
                    int left = int((cx - 0.5 * w) * scale);
                    int top = int((cy - 0.5 * h) * scale);
                    int width = int(w * scale);
                    int height = int(h * scale);
                    // Get the keypoints
                    std::vector<float> keypoints;
                    cv::Mat kpts = output_buffer.row(i).colRange(8, 23);
                    for (int i = 0; i < 5; i++) {
                        float x = kpts.at<float>(0, i * 3 + 0) * scale;
                        float y = kpts.at<float>(0, i * 3 + 1) * scale;
                        float s = kpts.at<float>(0, i * 3 + 2);
                        keypoints.push_back(x);
                        keypoints.push_back(y);
                        keypoints.push_back(s);
                    }

                    boxes.push_back(cv::Rect(left, top, width, height));
                    objects_keypoints.push_back(keypoints);

                    

                    }
                }

            // std::cout<<"boxes.size() is : "<<boxes.size()<<std::endl;

            //NMS
            std::vector<int> indices;
            if(boxes.size()>0){
                cv::dnn::NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);
            }

            // -------- Visualize the detection results -----------
            for (size_t i = 0; i < indices.size(); i++) {
                int index = indices[i];
                // Draw bounding box
                cv::rectangle(dst, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
                std::string label = std::to_string(class_id) + ":" + std::to_string(class_scores[index]).substr(0, 4);
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
                cv::Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
                cv::rectangle(dst, textBox, cv::Scalar(0, 0, 255), cv::FILLED);
                cv::putText(dst, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                // Draw keypoints
                std::vector<float> object_keypoints = objects_keypoints[index];
                for (int i = 0; i < 5; i++)
                {
                int x = std::clamp(int(object_keypoints[i * 3 + 0]), 0, dst.cols);
                int y = std::clamp(int(object_keypoints[i * 3 + 1]), 0, dst.rows);
                //Draw point
                cv::circle(dst, cv::Point(x, y), 5, cv::Scalar(0,255,0), -1);
                cv::putText(dst,std::to_string(i),cv::Point(x, y + 1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0),3);
                }
                // Draw keypoints-line

                //画出扇叶中心点
                cv::Point2f observed_point = leaf_center(dst,object_keypoints);

                // //提取ROI区域
                // roi = img1(boxes[index]);
                // // roi = dst(boxes[index]);
                // cv::imshow("roi",roi);



                //kalmen预测
                //预测方程
                X(1,0) = (observed_point.x - X1(0,0)) / t;     //计算x,y,方向的速度，把它们放在X矩阵中该放的位置
                X(3,0) = (observed_point.y - X1(1,0)) / t;     //计算方法用上面pnp解算出的tvec的坐标，x,y,z分别(新的x-上次循环的x) / t;

                X(0,0) = X(0,0) + X(1,0) * (t + t0);     //位置更新方程，原理就是 x = x0 + vt;
                X(2,0) = X(2,0) + X(3,0) * (t + t0);

                //将预测后的点放进X3                
                X3.at<double>(0,0) = X(0,0);
                X3.at<double>(1,0) = X(2,0);

                cv::circle(dst,cv::Point2f(X3.at<double>(0,0),X3.at<double>(1,0)),3,cv::Scalar(255,0,0),-1);

                cv::Point3f send_point(X3.at<double>(0,0),X3.at<double>(1,0),distance);

                // cv::Point2f aiming_point(X3.at<double>(0,0),X3.at<double>(1,0));
                // cv::circle(dst,aiming_point,3,cv::Scalar(0,255,0),-1);
                // float now_angle = 60;
                // float compensation_angle = TC(now_angle,aiming_point.y - center.y,aiming_point,dst);

                // std::cout<<0.5 * g * 2<<std::endl;


                F <<    1,  0.1,  0,  0,
                        0,  1,  0,  0,
                        0,  0,  1,  0.1,
                        0,  0,  0,  1;
                      

                variable1 = pow(t,4) / 4;
                variable2 = pow(t,2);
                variable3 = pow(t,3) / 2;
                Q <<    variable1,  variable3,          0,          0,
                        variable3,  variable2,          0,          0,
                                0,          0,  variable1,  variable3,
                                0,          0,  variable3,  variable2;


                Q = Q * a;

                P = F * P * F.transpose() + Q;

                //更新方程
                //Hnew1.3
                H <<    1,  0,  0,  0,
                        0,  0,  1,  0;
                
                //(R矩阵中的值待估计）
                R <<    0.001,  0,
                        0,  0.001;
                // R <<    0,  0,
                //         0,  0;
                        

                K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

                X2 << observed_point.x,observed_point.y;

                X = X + K * (X2 - H * X);

                P = (I - K * H) * P;

                X1(0,0) = observed_point.x;
                X1(1,0) = observed_point.y;

            }
        }

        
		// cv::Size shape = dst.size();
		// plot_keypoints(dst, objects_keypoints, shape);
		if (img1.empty())break;
		cv::imshow("dst",dst);
        // cv::imshow("roi",roi);
		// cv::waitKey(0);
		if (cv::waitKey(1) >= 0) break;


	}	

	capture.release();

	return 0;
}