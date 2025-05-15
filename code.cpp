#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
string img_path = "/Users/cothang/Documents/WorkPlace/CV/DATN-PARKING-LOT-DETECTING/parking.jpg";
Mat color_img, gray_img, edge, blur_img, otsu, edge_otsu, otsu_filtered;
vector<vector<Point> > contours;
RNG rng(12345);
float min_distance = 15; // Hai duong thang cach nhau 10 pixel xem nhu la 1
vector<Point> center_of_car;

float a1, b1, a2, b2;
int main(int argc, char** argv) {
    // Load the image
    color_img = cv::imread(img_path);
    gray_img = imread(img_path, 0);
  
    cv::GaussianBlur(gray_img, blur_img, Size(5, 5), 1, 1);
    
    cv::Canny(blur_img, edge, 100, 200); // for hough line
    
    cv::threshold(blur_img, otsu, 100, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    /*
        Hough line detect packing slot
        Step 1: Hough line with canny edge image
        Step 2:
    */

    Mat drawing = Mat::ones(edge.size(), CV_8UC1) * 255;
    // hough line find packing slot
    vector<Vec2f> lines; // will hold the results of the detection
    cv::HoughLines(edge, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
    
    // draw hough lines
    Mat hough_lines_img;
    cv::copyTo(color_img, hough_lines_img, hough_lines_img);
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(hough_lines_img, pt1, pt2, Scalar(0, 0, 255), 1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    // remove lines
    vector<Vec2f> intersection_point_with_xy_axis; // intersections of lines with x y axis
    vector<Vec2f> lines_vector; // vecto of line direction
    // find intersections of lines with x y axis
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        float a = cos(theta), b = sin(theta);
        float x0 = a * rho, y0 = b * rho;
        if (a != 0) {
            float t = -y0 / a;
            if (x0 - b * t >= 0) {
                intersection_point_with_xy_axis.push_back({ x0 - b * t, 0 }); //with x axis
            }
            else {
                float t = x0 / b;
                intersection_point_with_xy_axis.push_back({ 0, y0 + a * t }); //with y axis
            }
        }
        else {
            intersection_point_with_xy_axis.push_back({ 0, y0 });
        }
        lines_vector.push_back({ -b, a });
    }

    // sort coordinate from small to big
    for (size_t i = 0; i < intersection_point_with_xy_axis.size() - 1; i++) {
        for (size_t j = i+1; j < intersection_point_with_xy_axis.size(); j++) {

            if (intersection_point_with_xy_axis[j][0] < intersection_point_with_xy_axis[i][0]) {
                swap(intersection_point_with_xy_axis[i], intersection_point_with_xy_axis[j]);
                swap(lines_vector[i], lines_vector[j]);
            }
            if (intersection_point_with_xy_axis[j][0] == intersection_point_with_xy_axis[i][0] && intersection_point_with_xy_axis[j][1] < intersection_point_with_xy_axis[i][1]) {
                swap(intersection_point_with_xy_axis[i], intersection_point_with_xy_axis[j]);
                swap(lines_vector[i], lines_vector[j]);
            }
        }
    }

    // keep only lines with distance > min_distance
    vector<Vec2f> filtered_points, filtered_vector;
    int index_filtered_points = 0;
    filtered_points.push_back(intersection_point_with_xy_axis[0]);
    filtered_vector.push_back(lines_vector[0]);
    int nb_horizontal_lines = 0;
    if (filtered_points[0][0] == 0)
        nb_horizontal_lines += 1;
    
    for (size_t i = 1; i < intersection_point_with_xy_axis.size(); i++)
    {

        if (intersection_point_with_xy_axis[i][0] - filtered_points[index_filtered_points][0] > min_distance || intersection_point_with_xy_axis[i][1] - filtered_points[index_filtered_points][1] > min_distance) {
            filtered_points.push_back(intersection_point_with_xy_axis[i]);
            filtered_vector.push_back(lines_vector[i]);
            
            if (intersection_point_with_xy_axis[i][0] == 0)
                nb_horizontal_lines += 1;
            index_filtered_points += 1;
        }
    }
    ///////////////////////////////////////////////////////////
    Mat hough_lines_img_filtered;
    cv::copyTo(color_img, hough_lines_img_filtered, hough_lines_img_filtered);
    // draw lines 

    for (size_t i = 0; i < filtered_points.size(); i++)
    {
        Point pt1, pt2;
        float x1, y1, x2, y2;
        x1 = filtered_points[i][0] - 1000 * filtered_vector[i][0];
        x2 = filtered_points[i][0] + 1000 * filtered_vector[i][0];
        y1 = filtered_points[i][1] - 1000 * filtered_vector[i][1];
        y2 = filtered_points[i][1] + 1000 * filtered_vector[i][1];
        pt1.x = cvRound(x1);
        pt1.y = cvRound(y1);
        pt2.x = cvRound(x2);
        pt2.y = cvRound(y2);
        line(drawing, pt1, pt2, Scalar(0, 0, 255), 1);  
        line(hough_lines_img_filtered, pt1, pt2, Scalar(0, 0, 255), 1);
    }
    cv::imshow("filtered hough lines", hough_lines_img_filtered);
    
    /////////////////////////////////////////////////////////////////
    // find intersections between lines and horizontal border
    int nb_slots = (filtered_points.size() - nb_horizontal_lines)*(1+ nb_horizontal_lines);
    // find intersection of vertical line with top horizontal border
    vector<vector<Point>> all_intersection_points;
    vector<Point> temp;
   
    for (size_t i = nb_horizontal_lines; i < filtered_points.size(); i++) {
        Point its_point;
        its_point.x = filtered_points[i][0];
        its_point.y = filtered_points[i][1];
        temp.push_back(its_point);
   
    }

    all_intersection_points.push_back(temp);
    for (size_t i = 0; i < nb_horizontal_lines; i++) {
        temp.clear();
        float x1, y1, x2, y2;
        a1 = filtered_vector[i][1];
        b1 = filtered_vector[i][0];
        x1 = filtered_points[i][0];
        y1 = filtered_points[i][1];
        for (size_t j = nb_horizontal_lines; j < filtered_points.size(); j++) {
            
            a2 = filtered_vector[j][1];
            b2 = filtered_vector[j][0];
            x2 = filtered_points[j][0];
            y2 = filtered_points[j][1];

            float t2 = (b1 * (y1 - y2) - a1 * (x1 - x2)) / (a2 * b1 - a1 * b2);
            Point its_point;
            its_point.x = x2 + b2 * t2;
            its_point.y = y2 + a2 * t2;
            temp.push_back(its_point);

        }
        all_intersection_points.push_back(temp);
    }
    temp.clear();
    // find intersection of vertical line with below horizontal border
    float x1, y1, x2, y2;
    a1 = 0;
    b1 = 1;
    x1 = 0;
    y1 = gray_img.rows;
    for (size_t j = nb_horizontal_lines; j < filtered_points.size(); j++) {
     
        a2 = filtered_vector[j][1];
        b2 = filtered_vector[j][0];
        x2 = filtered_points[j][0];
        y2 = filtered_points[j][1];

        float t2 = (b1 * (y1 - y2) - a1 * (x1 - x2)) / (a2 * b1 - a1 * b2);
        Point its_point;
        its_point.x = x2 + b2 * t2;
        its_point.y = y2 + a2 * t2;
        temp.push_back(its_point);
    }
    all_intersection_points.push_back(temp);

    // draw parking slot
    Mat parking_slot;
    cv::copyTo(color_img, parking_slot, parking_slot);
    for (int j = 0; j < all_intersection_points.size() - 1; j++) {
        for (int k = 0; k < all_intersection_points[j].size() - 1; k++) {
            int is_existed = 0;
            Point tl, br;
            tl.x = all_intersection_points[j][k].x + 2;
            tl.y = all_intersection_points[j][k].y + 2;
            br.x = all_intersection_points[j + 1][k + 1].x - 2;
            br.y = all_intersection_points[j + 1][k + 1].y - 2;
            // check if j k is existed in vector?
            
            cv::rectangle(parking_slot, tl, br, Scalar(0, 0, 255), 2);
        }
    }
    cv::imshow("packing slot", parking_slot);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
        Find car location
        Step 1: noise filter for otsu img
        Step 2: canny edge detection for the result of step 1
        Step 3: find contours
        Step 4: find bounding rectangle of contours, only keep rectangles which have area bigger than 3500
    */
    // noise filter
    Mat kernel = Mat::ones(3, 3, CV_8UC1) * 255;
    cv::dilate(otsu, otsu_filtered, kernel);
    kernel = Mat::ones(7, 7, CV_8UC1) * 255;
    
    cv::erode(otsu_filtered, otsu_filtered, kernel);
    cv::Canny(otsu_filtered, edge_otsu, 100, 200); // for car location
    
    // find contours of edge otsu - draw rectangle bounding box of car
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edge_otsu, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
   
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    Mat rectangle_box_color;
    cv::copyTo(color_img, rectangle_box_color, rectangle_box_color);
    for (size_t i = 0; i < contours.size(); i++) {
        approxPolyDP(contours[i], contours_poly[i], 3, false);
        boundRect[i] = cv::boundingRect(contours_poly[i]);
        int h = boundRect[i].height;
        int w = boundRect[i].width;
        if (h * w > 3500) { // keep only bounding boxes of cars
            rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), (0, 0, 0), 1);
            rectangle(rectangle_box_color, boundRect[i].tl(), boundRect[i].br(), (0, 0, 255), 1);
            int c_x = w / 2 + boundRect[i].x;
            int c_y = h / 2 + boundRect[i].y;
            Point center;
            center.x = c_x;
            center.y = c_y;
            center_of_car.push_back(center);
            
        }
    }
    // find parked slot
    vector<Vec2i> jk_index_parked, jk_index_empty;
    for (size_t i = 0; i < center_of_car.size(); i++) {
        int cx = center_of_car[i].x;
        int cy = center_of_car[i].y;
        for (int j = 0; j < all_intersection_points.size()-1; j++) {
            for (int k = 0; k < all_intersection_points[j].size() - 1; k++) {
                float x_tl, x_tr, x_bl, x_br, y_tl, y_tr, y_bl, y_br;
                x_tl = all_intersection_points[j][k].x;
                x_tr = all_intersection_points[j][k+1].x;
                x_bl = all_intersection_points[j+1][k].x;
                x_br = all_intersection_points[j+1][k+1].x;

                y_tl = all_intersection_points[j][k].y;
                y_tr = all_intersection_points[j][k + 1].y;
                y_bl = all_intersection_points[j + 1][k].y;
                y_br = all_intersection_points[j + 1][k + 1].y;

                if (cx <= x_br && cx >= x_tl && cy >= y_tl && cy <= y_br) {
                    int is_existed = 0;
                    // check if j k is existed in vector?
                    if (jk_index_parked.size() > 0) {
                        for (int l = 0; l < jk_index_parked.size(); l++) {
                            if (jk_index_parked[l][0] == j && jk_index_parked[l][1] == k) {
                                is_existed = 1;
                                break;
                            }
                        }
                        if (is_existed == 0)
                            jk_index_parked.push_back({ j, k });
                    }
                    else
                        jk_index_parked.push_back({ j, k });
                }
                
            }
        }
    }
   

    // Draw 
    for (int j = 0; j < all_intersection_points.size() - 1; j++) {
        for (int k = 0; k < all_intersection_points[j].size() - 1; k++) {
            int is_existed = 0;
            Point tl, br;
            tl.x = all_intersection_points[j][k].x+2;
            tl.y = all_intersection_points[j][k].y+2;
            br.x = all_intersection_points[j+1][k+1].x-2;
            br.y = all_intersection_points[j+1][k+1].y-2;
            // check if j k is existed in vector?
            for (int l = 0; l < jk_index_parked.size(); l++) {
                if (jk_index_parked[l][0] == j && jk_index_parked[l][1] == k) {
                    is_existed = 1;
                    break;
                }
            }
            if (is_existed == 0) {
                cv::rectangle(color_img, tl, br, Scalar(0, 255, 0), 2);
                jk_index_empty.push_back({ j, k });
            }
            else
                cv::rectangle(color_img, tl, br, Scalar(0, 0, 255), 2);
        }
    }
    std::cout << "Parked position:" << endl;
    for (int i = 0; i < jk_index_parked.size(); i++)
        std::cout << jk_index_parked[i] << "\t";
    std::cout << "\nEmpty position:" << endl;
    for (int i = 0; i < jk_index_empty.size(); i++)
        std::cout << jk_index_empty[i] << "\t";

    cv::imshow("result of b", color_img);
    cv::waitKey(0);
    return 0;
}
