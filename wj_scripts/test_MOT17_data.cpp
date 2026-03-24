#include <motcpp/trackers/sort.hpp>
#include <motcpp/trackers/ucmc.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <motcpp/trackers/ocsort.hpp>
#include <motcpp/trackers/deepocsort.hpp>
#include <motcpp/trackers/strongsort.hpp>
#include <motcpp/trackers/botsort.hpp>
#include <motcpp/trackers/boosttrack.hpp>
#include <motcpp/trackers/hybridsort.hpp>
#include <motcpp/trackers/oracletrack.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
// #include <bits/stdc++.h>
#include <chrono>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

std::unordered_map<int, std::vector<std::array<float, 5>>> readFromDetectionFile(string detectionfilepath){
    // frame_id -> list of detections
    std::unordered_map<int, std::vector<std::array<float, 5>>> det_map;

    // Read det.txt
    std::ifstream file(detectionfilepath.c_str());
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> values;

        while (std::getline(ss, token, ',')) {
            values.push_back(std::stof(token));
        }

        int frame_id = static_cast<int>(values[0]);
        float left = values[2];
        float top = values[3];
        float width = values[4];
        float height = values[5];
        float conf = values[6];

        float x1 = left;
        float y1 = top;
        float x2 = left + width;
        float y2 = top + height;

        det_map[frame_id].push_back({x1, y1, x2, y2, conf});
    }

    return det_map;
}

int main(int argc, char* argv[]) {

    cout<<"in main"<<endl;
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0),     // Olive
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 0),      // Dark Blue
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(0, 0, 128),      // Dark Red
        cv::Scalar(128, 128, 0)     // Olive
    };

    // Create ByteTrack tracker with default parameters
    motcpp::trackers::ByteTrack tracker;

    // motcpp::trackers::HybridSort tracker(
    //     0.3f,   // det_thresh
    //     30,     // max_age
    //     50,     // max_obs
    //     3,      // min_hits
    //     0.3f,   // iou_threshold
    //     "hmiou" // asso_func: "iou" or "hmiou"
    // );

    std::string tracking_method = "bytetrack";
    std::string reid_weights = "";

    // Initialize tracker based on tracking_method
    // std::unique_ptr<motcpp::BaseTracker> tracker;
    // if (tracking_method == "sort") {
    //     // SORT - Original Simple Online and Realtime Tracking
    //     tracker = std::make_unique<motcpp::trackers::Sort>(
    //         0.3f,   // det_thresh
    //         1,      // max_age (original SORT uses 1)
    //         50,     // max_obs
    //         3,      // min_hits
    //         0.3f,   // iou_threshold
    //         false,  // per_class
    //         80,     // nr_classes
    //         "iou",  // asso_func
    //         false   // is_obb
    //     );
    // } else if (tracking_method == "ucmc") {
    //     // UCMCTrack - Unified Confidence-based Multi-object tracker
    //     tracker = std::make_unique<motcpp::trackers::UCMCTrack>(
    //         0.3f,   // det_thresh
    //         30,     // max_age
    //         50,     // max_obs
    //         3,      // min_hits
    //         0.3f,   // iou_threshold
    //         false,  // per_class
    //         80,     // nr_classes
    //         "iou",  // asso_func
    //         false,  // is_obb
    //         100.0,  // a1 - high-conf association threshold
    //         100.0,  // a2 - low-conf association threshold
    //         5.0,    // wx - process noise x
    //         5.0,    // wy - process noise y
    //         10.0,   // vmax - max velocity
    //         1.0 / seq_info.fps,  // dt - time step
    //         0.5f    // high_score - confidence split
    //     );
    // } else if (tracking_method == "bytetrack") {
    //     tracker = std::make_unique<motcpp::trackers::ByteTrack>(
    //         0.3f,   // det_thresh
    //         30,     // max_age
    //         50,     // max_obs
    //         3,      // min_hits
    //         0.3f,   // iou_threshold
    //         false,  // per_class
    //         80,     // nr_classes
    //         "iou",  // asso_func
    //         false,  // is_obb
    //         0.1f,   // min_conf
    //         0.45f,  // track_thresh
    //         0.8f,   // match_thresh
    //         30,     // track_buffer
    //         seq_info.fps  // frame_rate
    //     );
    // } else if (tracking_method == "ocsort") {
    //     tracker = std::make_unique<motcpp::trackers::OCSort>(
    //         0.2f,   // det_thresh (OCSort default)
    //         30,     // max_age
    //         50,     // max_obs
    //         3,      // min_hits
    //         0.3f,   // iou_threshold
    //         false,  // per_class
    //         80,     // nr_classes
    //         "iou",  // asso_func
    //         false,  // is_obb
    //         0.1f,   // min_conf
    //         3,      // delta_t
    //         0.2f,   // inertia
    //         false,  // use_byte
    //         0.01f,  // Q_xy_scaling
    //         0.0001f // Q_s_scaling
    //     );
    // } else if (tracking_method == "deepocsort") {
    //     if (reid_weights.empty()) {
    //         std::cerr << "Error: DeepOCSort requires reid_weights path (7th argument)\n";
    //         return 1;
    //     }
    //     tracker = std::make_unique<motcpp::trackers::DeepOCSort>(
    //         reid_weights,  // reid_weights
    //         false,        // use_half
    //         false,        // use_gpu
    //         0.3f,         // det_thresh
    //         30,           // max_age
    //         50,           // max_obs
    //         3,            // min_hits
    //         0.3f,         // iou_threshold
    //         false,        // per_class
    //         80,           // nr_classes
    //         "iou",        // asso_func
    //         false,        // is_obb
    //         3,            // delta_t
    //         0.2f,         // inertia
    //         0.5f,         // w_association_emb
    //         0.95f,        // alpha_fixed_emb
    //         0.5f,         // aw_param
    //         false,        // embedding_off
    //         false,        // cmc_off
    //         false,        // aw_off
    //         0.01f,        // Q_xy_scaling
    //         0.0001f       // Q_s_scaling
    //     );
    // } else if (tracking_method == "strongsort") {
    //     // ReID weights are optional - can use pre-generated embeddings instead
    //     // Default parameters for strongsort.yaml
    //     tracker = std::make_unique<motcpp::trackers::StrongSORT>(
    //         reid_weights,  // reid_weights
    //         false,         // use_half
    //         false,         // use_gpu
    //         0.3f,          // det_thresh
    //         30,            // max_age
    //         50,            // max_obs
    //         3,             // min_hits
    //         0.3f,          // iou_threshold
    //         false,         // per_class
    //         80,            // nr_classes
    //         "iou",         // asso_func
    //         false,         // is_obb
    //         0.6f,          // min_conf (Python default: 0.6)
    //         0.4f,          // max_cos_dist (Python default: 0.4)
    //         0.7f,          // max_iou_dist
    //         3,             // n_init
    //         100,           // nn_budget
    //         0.98f,         // mc_lambda
    //         0.9f           // ema_alpha
    //     );
    // } else if (tracking_method == "botsort") {
    //     // Default parameters for botsort.yaml
    //     tracker = std::make_unique<motcpp::trackers::BotSort>(
    //         reid_weights.empty() ? "" : reid_weights,  // reid_weights
    //         false,         // use_half
    //         false,         // use_gpu
    //         0.3f,          // det_thresh
    //         30,            // max_age (track_buffer)
    //         50,            // max_obs
    //         3,             // min_hits
    //         0.3f,          // iou_threshold
    //         false,         // per_class
    //         80,            // nr_classes
    //         "iou",         // asso_func
    //         false,         // is_obb
    //         0.6f,          // track_high_thresh (Python default: 0.6)
    //         0.1f,          // track_low_thresh (Python default: 0.1)
    //         0.7f,          // new_track_thresh (Python default: 0.7)
    //         30,            // track_buffer (Python default: 30)
    //         0.8f,          // match_thresh (Python default: 0.8)
    //         0.5f,          // proximity_thresh (Python default: 0.5)
    //         0.25f,         // appearance_thresh (Python default: 0.25)
    //         "ecc",         // cmc_method (Python default: ecc)
    //         seq_info.fps,  // frame_rate
    //         false,         // fuse_first_associate
    //         !reid_weights.empty()  // with_reid
    //     );
    // } else if (tracking_method == "boosttrack") {
    //     // Default parameters for boosttrack.yaml
    //     // BoostTrack++ uses use_rich_s=True, use_sb=True, use_vt=True
    //     tracker = std::make_unique<motcpp::trackers::BoostTrackTracker>(
    //         reid_weights.empty() ? "" : reid_weights,  // reid_weights
    //         false,         // use_half
    //         false,         // use_gpu
    //         0.6f,          // det_thresh (Python default: 0.6)
    //         60,            // max_age (Python default: 60)
    //         50,            // max_obs
    //         3,             // min_hits (Python default: 3)
    //         0.3f,          // iou_threshold (Python default: 0.3)
    //         false,         // per_class
    //         80,            // nr_classes
    //         "iou",         // asso_func
    //         false,         // is_obb
    //         true,          // use_ecc (Python default: True)
    //         10,            // min_box_area (Python default: 10)
    //         1.6f,          // aspect_ratio_thresh (Python default: 1.6)
    //         "ecc",         // cmc_method
    //         0.5f,          // lambda_iou (Python default: 0.5)
    //         0.25f,         // lambda_mhd (Python default: 0.25)
    //         0.25f,         // lambda_shape (Python default: 0.25)
    //         true,          // use_dlo_boost (Python default: True)
    //         true,          // use_duo_boost (Python default: True)
    //         0.65f,         // dlo_boost_coef (Python default: 0.65)
    //         false,         // s_sim_corr (Python default: False)
    //         true,          // use_rich_s (Python default: True for BoostTrack++)
    //         true,          // use_sb (Python default: True for BoostTrack++)
    //         true,          // use_vt (Python default: True for BoostTrack++)
    //         !reid_weights.empty()  // with_reid (Python default: True)
    //     );
    // } else if (tracking_method == "hybridsort") {
    //     // Default parameters for hybridsort.yaml
    //     tracker = std::make_unique<motcpp::trackers::HybridSort>(
    //         reid_weights.empty() ? "" : reid_weights,  // reid_weights
    //         false,         // use_half
    //         false,         // use_gpu
    //         0.5f,          // det_thresh (use track_thresh from Python: 0.5)
    //         30,            // max_age
    //         50,            // max_obs
    //         3,             // min_hits
    //         0.3f,          // iou_threshold (typical tracking threshold)
    //         false,         // per_class
    //         80,            // nr_classes
    //         "hmiou",       // asso_func
    //         false,         // is_obb
    //         0.1f,          // low_thresh (Python default: 0.1)
    //         3,             // delta_t (Python default: 3)
    //         0.05f,         // inertia (Python default: 0.05)
    //         true,          // use_byte (Python default: True)
    //         true,          // use_custom_kf (Python default: True)
    //         30,            // longterm_bank_length (Python default: 30)
    //         0.9f,          // alpha (Python default: 0.9)
    //         false,         // adapfs (Python default: False)
    //         0.5f,          // track_thresh (Python default: 0.5)
    //         4.6f,          // EG_weight_high_score (Python default: 4.6)
    //         1.3f,          // EG_weight_low_score (Python default: 1.3)
    //         true,          // TCM_first_step (Python default: True)
    //         true,          // TCM_byte_step (Python default: True)
    //         1.0f,          // TCM_byte_step_weight (Python default: 1.0)
    //         0.7f,          // high_score_matching_thresh (Python default: 0.7)
    //         true,          // with_longterm_reid (Python default: True)
    //         0.0f,          // longterm_reid_weight (Python default: 0.0)
    //         true,          // with_longterm_reid_correction (Python default: True)
    //         0.4f,          // longterm_reid_correction_thresh (Python default: 0.4)
    //         0.4f,          // longterm_reid_correction_thresh_low (Python default: 0.4)
    //         "ecc",         // cmc_method
    //         !reid_weights.empty()  // with_reid
    //     );
    // } else if (tracking_method == "oracletrack") {
    //     // OracleTrack - Novel tracker with proper Kalman filtering + cascaded association
    //     tracker = std::make_unique<motcpp::trackers::OracleTrack>(
    //         0.3f,   // det_thresh
    //         30,     // max_age (optimal for track recovery)
    //         3,      // min_hits (standard: require 3 hits before output)
    //         9.21f,  // gating_threshold (not used with IoU matching)
    //         4.0f    // max_mahalanobis (not used with IoU matching)
    //     );
    // } else {
    //     std::cerr << "Unknown tracking method: " << tracking_method << "\n";
    //     std::cerr << "Supported methods: sort, ucmc, bytetrack, ocsort, deepocsort, strongsort, botsort, boosttrack, hybridsort, oracletrack\n";
    //     return 1;
    // }
    
    std::string image_folder = "D:\\WJ_git\\motcpp\\data\\OpenDataLab___MOT17\\raw\\MOT17\\test\\MOT17-01-DPM\\img1\\";
    string detectionfilepath = "D:\\WJ_git\\motcpp\\data\\OpenDataLab___MOT17\\raw\\MOT17\\test\\MOT17-01-DPM\\det\\det.txt";
    std::vector<std::string> image_paths;

    std::unordered_map<int, std::vector<std::array<float, 5>>> det_map = readFromDetectionFile(detectionfilepath);

    // Collect only .jpg images
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.path().extension() == ".jpg") {
            image_paths.push_back(entry.path().string());
        }
    }

    // Zero-padded filenames → normal sort works
    std::sort(image_paths.begin(), image_paths.end());

    VideoWriter writer("output.mp4",VideoWriter::fourcc('m','p','4','v'), 30, Size(1920,1080));

    for (const auto& path : image_paths) {
        cv::Mat frame = cv::imread(path);
        fs::path p(path);              // convert string → filesystem path
        std::string filename = p.filename().string();   // "000001.jpg"
        std::string stem     = p.stem().string();       // "000001"
        int frame_idx = std::stoi(stem);   // 1

        auto& dets = det_map[frame_idx];
        Eigen::MatrixXf detections(dets.size(), 6);

        cout<<"detections: "<<dets.size()<<endl;

        for (size_t i = 0; i < dets.size(); ++i) {
            detections(i, 0) = dets[i][0];  // x1
            detections(i, 1) = dets[i][1];  // y1
            detections(i, 2) = dets[i][2];  // x2
            detections(i, 3) = dets[i][3];  // y2
            detections(i, 4) = dets[i][4];  // confidence
            detections(i, 5) = 0;           // class_id (pedestrian)

            
        }

        auto start=std::chrono::high_resolution_clock::now();
        Eigen::MatrixXf tracks = tracker.update(detections, frame);
        auto end=std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed=end-start;
        cout<<elapsed.count()<<"sec"<<endl;

        cout<<"tracks: "<<tracks.rows()<<endl;
        for (int i = 0; i < tracks.rows(); ++i) {
            std::cout << "  Track ID: " << static_cast<int>(tracks(i, 4))
                    << " at [" << tracks(i, 0) << ", " << tracks(i, 1)
                    << ", " << tracks(i, 2) << ", " << tracks(i, 3) << "]\n";

            float x1 = tracks(i, 0);
            float y1 = tracks(i, 1);
            float x2 = tracks(i, 2);
            float y2 = tracks(i, 3);
            int trk_id = static_cast<int>(tracks(i, 4));
            float conf = tracks(i, 5);

            cv::Rect box(
                static_cast<int>(x1),
                static_cast<int>(y1),
                static_cast<int>(x2 - x1),
                static_cast<int>(y2 - y1)
            );

            // Green box for detections
            cv::rectangle(frame, box, colors[trk_id], 2);

            // Draw confidence score
            cv::putText(frame,
                        cv::format("%d %.2f", trk_id, conf),
                        box.tl(),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors[trk_id],
                        1);

        }
        cout<<path<<endl;
        cv::imshow("1", frame);
        waitKey(1);

        writer.write(frame);
    }

    writer.release();

    
    return 0;
}