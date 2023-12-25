
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

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
// img1 = cv::imread(IMAGE_PATH);
cv::VideoCapture capture;


//用于求扇叶中心点
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


// void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape)
// {

//     int radius = 5;
//     bool drawLines = true;

//     if (keypoints.empty()) {
//         return;
//     }

//     std::vector<cv::Scalar> limbColorPalette;
//     std::vector<cv::Scalar> kptColorPalette;

//     for (int index : limbColorIndices) {
//         limbColorPalette.push_back(posePalette[index]);
//     }

//     for (int index : kptColorIndices) {
//         kptColorPalette.push_back(posePalette[index]);
//     }

//     for (const auto& keypoint : keypoints) {
//         bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
//         drawLines &= isPose;

//         // draw points
//         for (int i = 0; i < 17; i++) {
//             int idx = i * 3;
//             int x_coord = static_cast<int>(keypoint[idx]);
//             int y_coord = static_cast<int>(keypoint[idx + 1]);

//             if (x_coord % shape.width != 0 && y_coord % shape.height != 0) {
//                 if (keypoint.size() == 3) {
//                     float conf = keypoint[2];
//                     if (conf < 0.5) {
//                         continue;
//                     }
//                 }
//                 cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
//                     255);  // Default to red if not in pose mode
//                 cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
//             }
//         }
//         // draw lines
//         if (drawLines) {
//             for (int i = 0; i < skeleton.size(); i++) {
//                 const std::vector<int>& sk = skeleton[i];
//                 int idx1 = sk[0] - 1;
//                 int idx2 = sk[1] - 1;

//                 int idx1_x_pos = idx1 * 3;
//                 int idx2_x_pos = idx2 * 3;

//                 int x1 = static_cast<int>(keypoint[idx1_x_pos]);
//                 int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
//                 int x2 = static_cast<int>(keypoint[idx2_x_pos]);
//                 int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

//                 float conf1 = keypoint[idx1_x_pos + 2];
//                 float conf2 = keypoint[idx2_x_pos + 2];

//                 // Check confidence thresholds
//                 if (conf1 < 0.5 || conf2 < 0.5) {
//                     continue;
//                 }

//                 // Check if positions are within bounds
//                 if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
//                     x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
//                     continue;
//                 }

//                 // Draw a line between keypoints
//                 cv::Scalar color_limb = limbColorPalette[i];
//                 cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
//             }
//         }
//     }
// }


int main() {

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

    clock_t start_time = 0, end_time = 0;




	for(int i = 0; i < frames; i++)
	{
        // clock_t start_time, end_time;
		// start_time = clock();
        // end_time = clock();
		// std::cout << "Time:" << (double)(end_time - start_time) << "ms" << std::endl;

        end_time = clock();
        std::cout << "Time:" << (double)(end_time - start_time) << "ms" << std::endl;
		start_time = clock();



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
                leaf_center(dst,object_keypoints);

            }
        }

        
		// cv::Size shape = dst.size();
		// plot_keypoints(dst, objects_keypoints, shape);
		if (img1.empty())break;
		cv::imshow("dst",dst);
		// cv::waitKey(0);
		if (cv::waitKey(1) >= 0) break;


	}	

	capture.release();

	return 0;
}