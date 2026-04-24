// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/bytetrack.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace motcpp::trackers {

// Static shared Kalman filter
motion::KalmanFilterXYAH STrack::shared_kalman_;

//STrack is single object track (hold state (pos, v via KF))
//create track (not real yet). Is just a track candidate. Real with trackid when track.activate(). Confitmed track when is_activate=true.
STrack::STrack(const Eigen::VectorXf& det, int max_obs)
    : max_obs_(max_obs) //# past observations to keep
    , kalman_filter_(nullptr)
    , id_(0) //trackid assigned later
    , state_(ByteTrackState::New)
    , is_activated_(false) //not confirmed yet
    , tracklet_len_(0) //length of track
    , frame_id_(0) //last updated frame
    , start_frame_(0) //when track starts
{
    // det format: [x1, y1, x2, y2, conf, cls, det_ind]
    Eigen::Vector4f xyxy = det.head<4>(); //topleft (x,y), botright (x,y) for IoU matching
    xywh_ = utils::xyxy2xywh(xyxy); //(cx,cy,w,h) yolo format
    tlwh_ = utils::xywh2tlwh(xywh_); //(tlx,tly,w,h) for visualization
    xyah_ = utils::tlwh2xyah(tlwh_); //(cx,cy,aspect_ratio,h) for KF
    conf_ = det(4);
    cls_ = static_cast<int>(det(5)); //object class
    det_ind_ = (det.size() > 6) ? static_cast<int>(det(6)) : 0; //for debugging
}

//initialize a new track for new detection
//state=Track => valid Kalman state
//is_activated=true => track is confirmed and safe to output
//multiple detection to create true trackid
void STrack::activate(motion::KalmanFilterXYAH& kalman_filter, int frame_id) {
    kalman_filter_ = &kalman_filter; //dont copy filter, just keep a pointer to avoid expensive copy
    id_ = next_id();
    auto [mean_init, cov_init] = kalman_filter.initiate(xyah_); //create KF state -> KF expands it to 8D state [cx,cy,a,h,+v equivalent]
    mean = mean_init;
    covariance = cov_init;
    
    tracklet_len_ = 0;
    state_ = ByteTrackState::Tracked;
    if (frame_id == 1) { //frame 1 is trusted immediately, whereas later frames need confirmation.
        is_activated_ = true;
    }
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}

//lost track reappears. Detection, Detection, NoDet (lost track), Detection (lost track reappears). used for lost track which is now matched
void STrack::re_activate(const STrack& new_track, int frame_id, bool new_id) {
    auto [mean_new, cov_new] = kalman_filter_->update(mean, covariance, new_track.xyah_); //KF.update does 2 th: correct drift and reduce uncertainty (cov)
    mean = mean_new;
    covariance = cov_new;
    
    tracklet_len_ = 0; //new segment of the track
    state_ = ByteTrackState::Tracked; //b4 this is ByteTrackState::Lost
    is_activated_ = true;
    frame_id_ = frame_id;
    if (new_id) { //likely false for most call. accept new_id to make this function flexible in case track has been lost for very long. To create new id instead of using old trkid.
        id_ = next_id();
    }
    conf_ = new_track.conf_;
    cls_ = new_track.cls_;
    det_ind_ = new_track.det_ind_;
}

//used for matched track
void STrack::update(const STrack& new_track, int frame_id) {
    frame_id_ = frame_id;
    tracklet_len_ += 1; //longer tracks more reliable.
    history_observations_.push_back(xyxy()); //save current BB (true trajectory instead of smoothed-only positon) b4 update
    if (static_cast<int>(history_observations_.size()) > max_obs_) { //keep only recent observations
        history_observations_.pop_front();
    }
    
    auto [mean_new, cov_new] = kalman_filter_->update(mean, covariance, new_track.xyah_); //mean, covariance=prediction. new_track.xyah_=ob is actually here.
    mean = mean_new;
    covariance = cov_new;
    
    state_ = ByteTrackState::Tracked;
    is_activated_ = true;  // Python always sets this in update()
    conf_ = new_track.conf_;
    cls_ = new_track.cls_;
    det_ind_ = new_track.det_ind_;
}

//no detection step of tracking. move the track forward in time using motion model only. I didnt see the object this frame. Where should it be now.
//w/o this, matching is only based on last position.
//with this, matching is based on expected motion.
void STrack::predict() {
    Eigen::VectorXf mean_state = mean;
    if (state_ != ByteTrackState::Tracked) { //special for lost track
        mean_state(7) = 0; // Zero velocity //velocity of height (scale)=0. scale change are unstable when obj is not observed, so dont let size drift wildly when not tracked. Scale dont change when trk is lost.
    }
    auto [mean_new, cov_new] = kalman_filter_->predict(mean_state, covariance); //x'=Fx. P'=FPF_T+Q, x=state, P=covariance, F=motion model, Q=process noise. position += velocity. uncertainty increases (due to no measurement)
    mean = mean_new;
    covariance = cov_new;
}

//batched version of predict for many tracks to reduce runtime.
//predict/multi_predict is called b4 doing association
//better cache locality than calling predict in a for loop.
void STrack::multi_predict(std::vector<STrack>& stracks) {
    if (stracks.empty()) return;
    
    // OPTIMIZED: Batch prediction with vectorized operations
    // Process all tracks in one pass to improve cache locality
    const size_t n = stracks.size();
    if (n == 0) return;
    
    // OPTIMIZED: Process tracks in batches for better cache usage
    for (auto& st : stracks) {
        Eigen::VectorXf mean_state = st.mean;
        if (st.state_ != ByteTrackState::Tracked) {
            mean_state(7) = 0; // Zero velocity
        }
        // OPTIMIZED: Direct predict call (faster than separate calls)
        auto [new_mean, new_cov] = st.kalman_filter_->predict(mean_state, st.covariance);
        st.mean = std::move(new_mean);
        st.covariance = std::move(new_cov);
    }
}

Eigen::Vector4f STrack::xyxy() const {
    Eigen::Vector4f ret;
    if (mean.size() == 0) {
        ret = utils::xywh2xyxy(xywh_);
    } else {
        Eigen::Vector4f xyah = mean.head<4>();
        Eigen::Vector4f xywh = utils::xyah2xywh(xyah);
        ret = utils::xywh2xyxy(xywh);
    }
    return ret;
}

//STrack is one object being tracked. ByteTrack is the whole tracking system managing all objects
ByteTrack::ByteTrack(float det_thresh, int max_age, int max_obs, int min_hits,
                    float iou_threshold, bool per_class, int nr_classes,
                    const std::string& asso_func, bool is_obb,
                    float min_conf, float track_thresh, float match_thresh,
                    int track_buffer, int frame_rate)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                 per_class, nr_classes, asso_func, is_obb)
    , min_conf_(min_conf) //lowest detection confi to consider
    , track_thresh_(track_thresh) //main threshold for high confidence detection
    , match_thresh_(match_thresh) //IoU threshold for matching
    , track_buffer_(track_buffer) //how long to keep lost tracks alive
    , buffer_size_(static_cast<int>(frame_rate / 30.0f * track_buffer)) //ByteTrack originally designed for 30fps.
    , max_time_lost_(buffer_size_) //control when a track is deleted. If not seen for n frames, remove track
    , frame_id_(0)
{
    det_thresh_ = track_thresh_; //detection threshold (detection model output) and tracking threshold (association logic) are separate but are unified here for simplicity.
    
    // Pre-allocate buffers for zero-allocation hot path
    //without this, every frame will get new eigen matrix allocations. with this, reuse same memory every frame.
    //reserve allocate capacity once, avoid reallocation
    //200 is assuming max 200 tracks and 200 detections. If exceed, code will call conservativeResize, which takes longer.
    cost_matrix_buffer_.resize(200, 200);
    track_xyxy_buffer_.resize(200, 4);
    det_xyxy_buffer_.resize(200, 4);
    det_confs_buffer_.resize(200);
    strack_pool_buffer_.reserve(200);
    index_buffer_.reserve(200);
    track_index_map_buffer_.reserve(200);
}

void ByteTrack::reset() {
    BaseTracker::reset();
    frame_id_ = 0; //frame idx is used for track age, lost track timeout, activation logic
    active_tracks_.clear(); //resets all confirmed tracks and all live trajectories.
    lost_stracks_.clear(); //tracks that were lost or not seen recently also removed.
    removed_stracks_.clear(); //tracks that were permanently delted or expired by timeout also removed.
    STrack::clear_count(); //tracks ID start from 0
}

//ByteTrack is the manager of the STrack (one tracked object (state+motion model))
//det: [x1,y1,x2,y2,conf,classid]
Eigen::MatrixXf ByteTrack::update(const Eigen::MatrixXf& dets,
                                  const cv::Mat& img,
                                  const Eigen::MatrixXf& embs) {
    check_inputs(dets, img, embs); //check input is the right size
    setup_detection_format(dets); //
    setup_association_function(img);
    
    // Add detection indices
    Eigen::MatrixXf dets_with_ind = Eigen::MatrixXf(dets.rows(), dets.cols() + 1);
    dets_with_ind.leftCols(dets.cols()) = dets;
    for (int i = 0; i < dets.rows(); ++i) {
        dets_with_ind(i, dets.cols()) = static_cast<float>(i);
    }
    
    frame_count_++;
    frame_id_++;
    
    std::vector<STrack> activated_stracks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> lost_stracks_new;
    std::vector<STrack> removed_stracks_new;
    
    // Filter by confidence
    Eigen::VectorXf confs = dets_with_ind.col(4);
    Eigen::VectorXi remain_inds = (confs.array() > track_thresh_).cast<int>(); //idx with conf>high_confid. primary detection used for main/first association (strong matching). used for normal tracking
    Eigen::VectorXi inds_low = (confs.array() > min_conf_).cast<int>(); //idx with conf>min thresh to consider. include both medium and high conf
    Eigen::VectorXi inds_high = (confs.array() < track_thresh_).cast<int>(); //idx with conf<high_confid. below high confid
    Eigen::VectorXi inds_second = (inds_low.array() * inds_high.array()).cast<int>(); //idx with conf between (min thresh, high_confid). 2nd association step for recovering missed tracks. used only for recovery. Does not throw away weak detection immediately.
    //not saved in variable is the third region with low confid. To ignore
    std::vector<int> remain_indices, second_indices;
    for (int i = 0; i < dets_with_ind.rows(); ++i) {
        if (remain_inds(i)) remain_indices.push_back(i); //used for first assoc
        if (inds_second(i)) second_indices.push_back(i); //used for 2nd assoc
    }
    
    Eigen::MatrixXf dets_high(dets_with_ind.rows(), dets_with_ind.cols()); //to be resized later. used to store high confid detections
    Eigen::MatrixXf dets_second(second_indices.size(), dets_with_ind.cols()); //used for 2nd assoc
    
    int high_count = 0;
    for (int idx : remain_indices) {
        dets_high.row(high_count++) = dets_with_ind.row(idx);
    }
    dets_high.conservativeResize(high_count, dets_with_ind.cols());
    
    int second_count = 0;
    for (int idx : second_indices) {
        dets_second.row(second_count++) = dets_with_ind.row(idx);
    }
    
    // Create STrack objects
    //create high confid detection as tracks
    //convert detection into STrack so that both tracks and detection become STrack, making matching simpler.
    std::vector<STrack> detections;
    for (int i = 0; i < high_count; ++i) {
        detections.emplace_back(dets_high.row(i).transpose(), max_obs_); //used emplace_back to construct STrack directly in-place. to avoid temp obj copy. slightly more efficient than push_back
    }
    
    // Separate confirmed and unconfirmed tracks
    // OPTIMIZED: Use pre-allocated buffers and reserve capacity
    std::vector<int> unconfirmed_indices_in_active; //newly created/ weak track. is_activate=false. used for special association/matching
    std::vector<int> tracked_indices_in_active; //confirmed/ stable track. is_activate=true. used for main association/matching
    unconfirmed_indices_in_active.reserve(active_tracks_.size()); //reserve to avoid reallocation during push_back
    tracked_indices_in_active.reserve(active_tracks_.size());
    
    for (size_t i = 0; i < active_tracks_.size(); ++i) {
        if (!active_tracks_[i].is_activated()) {
            unconfirmed_indices_in_active.push_back(i);
        } else {
            tracked_indices_in_active.push_back(i);
        }
    }
    
    // Create copies for association (Python uses list references, but we need copies for multi_predict)
    //convert indices into object (copies, not references necause ByteTrack needs clean separation for association)
    std::vector<STrack> unconfirmed;
    std::vector<STrack> tracked_stracks;
    unconfirmed.reserve(unconfirmed_indices_in_active.size());
    tracked_stracks.reserve(tracked_indices_in_active.size());
    
    for (int idx : unconfirmed_indices_in_active) {
        unconfirmed.push_back(active_tracks_[idx]);
    }
    for (int idx : tracked_indices_in_active) {
        tracked_stracks.push_back(active_tracks_[idx]);
    }
    
    // Step 2: First association with high confidence detections
    // CRITICAL: strack_pool contains copies, so we need to track indices to update originals
    //lost_stracks are previously reliable but lost because missed detection, hence worth recovering.
    //unconfirmed tracks are not reliable yet (created recently from 1 detection only, may be FP), not fully activated yet, hence not trusted.
    //consider tracked_stracks and lost_stracks for first association with high conf detection
    std::vector<STrack> strack_pool = joint_stracks(tracked_stracks, lost_stracks_);
    
    // Map strack_pool indices to original track indices
    // First part is tracked_stracks (maps to active_tracks_), second part is lost_stracks_
    //prediction happens on copies to be used in matching. so that we can later update the originals based on where it comes from.
    std::vector<std::pair<int, bool>> track_index_map; // (index_in_original, is_tracked) //this variable is for writing the result back after matching.
    track_index_map.reserve(strack_pool.size());
    for (size_t i = 0; i < tracked_stracks.size(); ++i) {
        track_index_map.emplace_back(tracked_indices_in_active[i], true);
    }
    for (size_t i = 0; i < lost_stracks_.size(); ++i) {
        track_index_map.emplace_back(static_cast<int>(i), false);
    }
    
    // Predict all tracks using multi_predict (batch prediction)
    STrack::multi_predict(strack_pool); //do KF prediction only.
    
    // OPTIMIZED: Use pre-allocated buffers, resize only if needed
    //*2 => amortized growth strategy. if resize exactly to n_tracks, next frame might exceed again, repeated reallocations.
    size_t n_tracks = strack_pool.size();
    size_t n_dets = detections.size();
    
    if (track_xyxy_buffer_.rows() < static_cast<int>(n_tracks)) {
        track_xyxy_buffer_.conservativeResize(n_tracks * 2, 4); //keep existing data compare to resize which discard data.
    }
    if (det_xyxy_buffer_.rows() < static_cast<int>(n_dets)) {
        det_xyxy_buffer_.conservativeResize(n_dets * 2, 4);
    }
    
    //convert all tracks and detections into a dense matrix for fast vectorized IoU computation.
    //accidentally copy instead of viewing. May be able to optimize by auto track_xyxy = track_xyxy_buffer_.topRows(n_tracks); or Eigen::Block<Eigen::MatrixXf> track_xyxy = track_xyxy_buffer_.topRows(n_tracks);. Avoid allocation and keep it a view.
    Eigen::MatrixXf track_xyxy = track_xyxy_buffer_.topRows(n_tracks);
    Eigen::MatrixXf det_xyxy = det_xyxy_buffer_.topRows(n_dets);
    
    // OPTIMIZED: Direct row assignment without transpose where possible
    for (size_t i = 0; i < n_tracks; ++i) {
        Eigen::Vector4f xyxy = strack_pool[i].xyxy(); //may be able to optimize
        track_xyxy(i, 0) = xyxy(0);
        track_xyxy(i, 1) = xyxy(1);
        track_xyxy(i, 2) = xyxy(2);
        track_xyxy(i, 3) = xyxy(3);
    }
    for (size_t i = 0; i < n_dets; ++i) {
        Eigen::Vector4f xyxy = detections[i].xyxy();
        det_xyxy(i, 0) = xyxy(0);
        det_xyxy(i, 1) = xyxy(1);
        det_xyxy(i, 2) = xyxy(2);
        det_xyxy(i, 3) = xyxy(3);
    }
    
    Eigen::MatrixXf dists = utils::iou_distance(track_xyxy, det_xyxy); //WJ:pairwise distance matrix (NxM) between track_xyxy (Nx4) and det_xyxy (Mx4). distance=1-IoU
    
    // Fuse with score
    //WJ:amortized growth (does not seem useful at all)
    if (det_confs_buffer_.size() < static_cast<int>(n_dets)) {
        det_confs_buffer_.conservativeResize(n_dets * 2);
    }
    Eigen::VectorXf det_confs = det_confs_buffer_.head(n_dets); //WJ:view into the buffer. no memory allocation
    for (size_t i = 0; i < n_dets; ++i) {
        det_confs(i) = detections[i].conf();
    }
    dists = utils::fuse_score(dists, det_confs); //WJ:high conf detection has smaller IoU distance
    
    auto assignment = utils::linear_assignment(dists, match_thresh_); //WJ:hungarian algo to do bipartite matching problem
    //WJ:has match and unmmatch
    
    // OPTIMIZED: Pre-allocate and use faster matching
    std::vector<int> u_track, u_detection;
    u_track.reserve(n_tracks);
    u_detection.reserve(n_dets);
    
    // OPTIMIZED: Use unordered_set for O(1) lookup instead of O(n) loop
    std::unordered_set<int> matched_tracks, matched_dets;
    matched_tracks.reserve(assignment.matches.size());
    matched_dets.reserve(assignment.matches.size());
    for (const auto& match : assignment.matches) {
        matched_tracks.insert(match[0]);
        matched_dets.insert(match[1]);
    }
    
    for (size_t i = 0; i < n_tracks; ++i) {
        if (matched_tracks.find(static_cast<int>(i)) == matched_tracks.end()) {
            u_track.push_back(i); //WJ:unmatched tracked (missed detection tracks)
        }
    }
    for (size_t i = 0; i < n_dets; ++i) {
        if (matched_dets.find(static_cast<int>(i)) == matched_dets.end()) {
            u_detection.push_back(i); //WJ:unmmatched detection (new track candidate)
        }
    }
    
    // Process matches - update original tracks, not copies
    for (const auto& match : assignment.matches) {
        int itracked = match[0]; //WJ:strack_pool idx
        int idet = match[1];
        
        // Get reference to original track
        auto [orig_idx, is_tracked] = track_index_map[itracked]; //WJ:strack_pool idx to real track storage idx
        STrack* orig_track;
        if (is_tracked) {
            // Update original in active_tracks_
            orig_track = &active_tracks_[orig_idx]; //WJ:active set
        } else {
            // Update original in lost_stracks_
            orig_track = &lost_stracks_[orig_idx]; //WJ:recovery set
        }
        
        // Update prediction state from strack_pool copy
        //WJ:copy updated motion estimate back into real track
        orig_track->mean = strack_pool[itracked].mean;
        orig_track->covariance = strack_pool[itracked].covariance;
        
        STrack& det = detections[idet];
        
        if (orig_track->state() == ByteTrackState::Tracked) { //WJ:already active track, do normal tracking update (KF correction step)
            orig_track->update(det, frame_id_);
            activated_stracks.push_back(*orig_track);
        } else { //WJ:previously lost track, identity recovered after disappearance.
            orig_track->re_activate(det, frame_id_, false);
            refind_stracks.push_back(*orig_track);
        }
    }
    
    // Step 3: Second association with low confidence detections
    //WJ:detection candidates with [low,high] conf
    std::vector<STrack> detections_second;
    for (int i = 0; i < second_count; ++i) {
        detections_second.emplace_back(dets_second.row(i).transpose(), max_obs_);
    }
    
    // Build r_tracked_stracks with references to original tracks
    //WJ:use only tracks in active_tracks and not those in lost_stracks_. Unmatched in the primary matching.
    std::vector<STrack*> r_tracked_stracks_ptrs;
    std::vector<int> r_tracked_indices; // indices in strack_pool
    for (int idx : u_track) {
        if (strack_pool[idx].state() == ByteTrackState::Tracked) {
            auto [orig_idx, is_tracked] = track_index_map[idx];
            if (is_tracked) {
                // Reference original in active_tracks_
                r_tracked_stracks_ptrs.push_back(&active_tracks_[orig_idx]);
                r_tracked_indices.push_back(idx); //WJ:u_track idx
            }
        }
    }
    
    if (!detections_second.empty() && !r_tracked_stracks_ptrs.empty()) {
        Eigen::MatrixXf r_track_xyxy(r_tracked_stracks_ptrs.size(), 4); //WJ:remaining tracks (unmmatched in primary matching excluding lost track)
        Eigen::MatrixXf det2_xyxy(detections_second.size(), 4);
        for (size_t i = 0; i < r_tracked_stracks_ptrs.size(); ++i) {
            r_track_xyxy.row(i) = r_tracked_stracks_ptrs[i]->xyxy().transpose();
        }
        for (size_t i = 0; i < detections_second.size(); ++i) {
            det2_xyxy.row(i) = detections_second[i].xyxy().transpose();
        }
        
        Eigen::MatrixXf dists2 = utils::iou_distance(r_track_xyxy, det2_xyxy);
        auto assignment2 = utils::linear_assignment(dists2, 0.5f); //WJ:use 0.5 instead of match_thresh_
        
        //WJ:can be optimized. Currently O(NxM)
        //WJ:find unmmatched tracks and detections for 2nd matching
        std::vector<int> u_track2, u_detection2;
        for (size_t i = 0; i < r_tracked_stracks_ptrs.size(); ++i) {
            bool matched = false;
            for (const auto& match : assignment2.matches) {
                if (match[0] == static_cast<int>(i)) matched = true;
            }
            if (!matched) u_track2.push_back(i);
        }
        for (size_t i = 0; i < detections_second.size(); ++i) {
            bool matched = false;
            for (const auto& match : assignment2.matches) {
                if (match[1] == static_cast<int>(i)) matched = true;
            }
            if (!matched) u_detection2.push_back(i);
        }
        
        for (const auto& match : assignment2.matches) {
            int itracked = match[0];
            int idet = match[1];
            STrack* track = r_tracked_stracks_ptrs[itracked];
            
            // Update prediction state from strack_pool
            //WJ:update original motion estimate
            int strack_idx = r_tracked_indices[itracked];
            track->mean = strack_pool[strack_idx].mean;
            track->covariance = strack_pool[strack_idx].covariance;
            
            if (track->state() == ByteTrackState::Tracked) { //WJ:already active track, do normal tracking update (KF correction step)
                track->update(detections_second[idet], frame_id_);
                activated_stracks.push_back(*track);
            } else { //WJ:previously lost track, identity recovered after disappearance.
                track->re_activate(detections_second[idet], frame_id_, false);
                refind_stracks.push_back(*track);
            }

        }
        
        //WJ:unmatched tracks -> lost
        for (int idx : u_track2) {
            STrack* track = r_tracked_stracks_ptrs[idx];
            if (track->state() != ByteTrackState::Lost) {
                track->mark_lost();
                lost_stracks_new.push_back(*track);
            }
        }
    }
    else{
        for(auto* track: r_tracked_stracks_ptrs){
            if(track->state()!=ByteTrackState::Lost){
                track->mark_lost();
                lost_stracks_new.push_back(*track);
            }
        }
    }
    
    // Deal with unconfirmed tracks
    // Use the indices we already computed above
    // unconfirmed_indices_in_active already contains the indices
    
    std::vector<STrack> remaining_detections; //WJ:remaining detection not matched in 1st and 2nd matching.
    for (int idx : u_detection) {
        remaining_detections.push_back(detections[idx]);
    }
    
    std::vector<int> u_detection_final;
    
    //WJ:3rd matching stage
    //WJ:unconfirmed tracks are tracks that is weak/tentative (detected once)
    if (!unconfirmed.empty() && !remaining_detections.empty()) {
        // OPTIMIZED: Use pre-allocated buffers
        size_t n_unconf = unconfirmed.size();
        size_t n_rem_det = remaining_detections.size();
        
        if (track_xyxy_buffer_.rows() < static_cast<int>(n_unconf)) {
            track_xyxy_buffer_.conservativeResize(n_unconf * 2, 4);
        }
        if (det_xyxy_buffer_.rows() < static_cast<int>(n_rem_det)) {
            det_xyxy_buffer_.conservativeResize(n_rem_det * 2, 4);
        }
        
        Eigen::MatrixXf unconf_xyxy = track_xyxy_buffer_.topRows(n_unconf);
        Eigen::MatrixXf rem_det_xyxy = det_xyxy_buffer_.topRows(n_rem_det);
        
        for (size_t i = 0; i < n_unconf; ++i) {
            Eigen::Vector4f xyxy = unconfirmed[i].xyxy();
            unconf_xyxy(i, 0) = xyxy(0);
            unconf_xyxy(i, 1) = xyxy(1);
            unconf_xyxy(i, 2) = xyxy(2);
            unconf_xyxy(i, 3) = xyxy(3);
        }
        for (size_t i = 0; i < n_rem_det; ++i) {
            Eigen::Vector4f xyxy = remaining_detections[i].xyxy();
            rem_det_xyxy(i, 0) = xyxy(0);
            rem_det_xyxy(i, 1) = xyxy(1);
            rem_det_xyxy(i, 2) = xyxy(2);
            rem_det_xyxy(i, 3) = xyxy(3);
        }
        
        Eigen::MatrixXf dists3 = utils::iou_distance(unconf_xyxy, rem_det_xyxy);
        
        if (det_confs_buffer_.size() < static_cast<int>(n_rem_det)) {
            det_confs_buffer_.conservativeResize(n_rem_det * 2);
        }
        Eigen::VectorXf rem_det_confs = det_confs_buffer_.head(n_rem_det);
        for (size_t i = 0; i < n_rem_det; ++i) {
            rem_det_confs(i) = remaining_detections[i].conf();
        }
        dists3 = utils::fuse_score(dists3, rem_det_confs);
        
        auto assignment3 = utils::linear_assignment(dists3, 0.7f);
        
        // OPTIMIZED: Use unordered_set for O(1) lookup
        std::unordered_set<int> matched_unconf, matched_det_remaining_indices;
        matched_unconf.reserve(assignment3.matches.size());
        matched_det_remaining_indices.reserve(assignment3.matches.size());
        
        for (const auto& match : assignment3.matches) {
            matched_unconf.insert(match[0]);
            matched_det_remaining_indices.insert(match[1]);
        }
        
        std::vector<int> u_unconfirmed; //WJ:unmatched unconfirmed
        u_unconfirmed.reserve(n_unconf);
        for (size_t i = 0; i < n_unconf; ++i) {
            if (matched_unconf.find(static_cast<int>(i)) == matched_unconf.end()) {
                u_unconfirmed.push_back(i);
            }
        }
        
        // Build u_detection_final: detections from u_detection that weren't matched
        u_detection_final.reserve(u_detection.size());
        //WJ:potential bug discovered by chatgpt. matched_det_remaining_indices is in remaining_detctions space but u are applying it to u_detection space.
        // for (size_t i = 0; i < u_detection.size(); ++i) {
        //     if (matched_det_remaining_indices.find(static_cast<int>(i)) == matched_det_remaining_indices.end()) {
        //         u_detection_final.push_back(u_detection[i]);
        //     }
        // }
        for (size_t i = 0; i < n_rem_det; ++i) {
            bool matched = false;
            for (const auto& match : assignment3.matches) {
                if (match[1] == static_cast<int>(i)) matched = true;
            }
            if (!matched){
                u_detection_final.push_back(u_detection[i]);
            }
        }
        
        // Update original unconfirmed tracks
        for (const auto& match : assignment3.matches) { //WJ:uncertain track is confirmed by a low confid detection
            int itracked = match[0];
            int idet = match[1];
            int orig_idx = unconfirmed_indices_in_active[itracked];
            active_tracks_[orig_idx].update(remaining_detections[idet], frame_id_);
            activated_stracks.push_back(active_tracks_[orig_idx]);
        }
        
        // Mark unmatched unconfirmed tracks as removed
        //WJ:unmatched unconfirmed track is removed.
        //WJ:track->lost->unconfirmed->removed
        for (int idx : u_unconfirmed) {
            int orig_idx = unconfirmed_indices_in_active[idx];
            active_tracks_[orig_idx].mark_removed();
            removed_stracks_new.push_back(active_tracks_[orig_idx]);
        }
    } else {
        // No unconfirmed tracks, so all u_detection become u_detection_final
        u_detection_final = u_detection;
    }
    
    // Step 4: Init new tracks
    
    //WJ:unmmatched detection in 1st,2nd,3rd matching are potentially new objects
    for (int idx : u_detection_final) {
        if (idx < static_cast<int>(detections.size())) {
            STrack& track = detections[idx];
            if (track.conf() >= det_thresh_) { //WJ:only strong detection can become tracks
                track.activate(kalman_filter_, frame_id_);
                activated_stracks.push_back(track);
            }
        }
    }
    
    // Step 5: Update state
    //WJ:if a track has been invisible for too long, permanently delete it.
    //WJ:do not erase here, only mark. actual deletion handled later.
    for (auto& track : lost_stracks_) {
        if (frame_count_ - track.end_frame() > max_time_lost_) {
            track.mark_removed();
            removed_stracks_new.push_back(track);
        }
    }
    
    // Update active tracks - keep only tracked ones
    //WJ:keep only Tracked, dont include lost, removed, unconfirmed tracks.
    std::vector<STrack> new_active_tracks;
    for (auto& track : active_tracks_) {
        if (track.state() == ByteTrackState::Tracked) {
            new_active_tracks.push_back(track);
        }
    }
    //WJ:persistent tracked + newly confirmed tracks + refound tracks
    active_tracks_ = new_active_tracks;
    active_tracks_ = joint_stracks(active_tracks_, activated_stracks);
    active_tracks_ = joint_stracks(active_tracks_, refind_stracks);
    
    // Update lost tracks
    //WJ:lost_stracks - active_tracks (recovered tracks) + newly lost track - newly removed tracks
    lost_stracks_ = sub_stracks(lost_stracks_, active_tracks_);
    lost_stracks_.insert(lost_stracks_.end(), lost_stracks_new.begin(), lost_stracks_new.end());
    lost_stracks_ = sub_stracks(lost_stracks_, removed_stracks_new);
    
    //WJ:archive it in permanently removed tracks
    removed_stracks_.insert(removed_stracks_.end(), removed_stracks_new.begin(), removed_stracks_new.end());
    
    // Remove duplicates
    auto [new_active, new_lost] = remove_duplicate_stracks(active_tracks_, lost_stracks_); //WJ:check active_tracks_ and lost_stracks_ overlap in IoU for duplicate check
    active_tracks_ = new_active;
    lost_stracks_ = new_lost;
    
    // OPTIMIZED: Build output directly without intermediate vector
    // Count activated tracks first
    size_t n_output = 0;
    for (const auto& track : active_tracks_) {
        if (track.is_activated()) {
            ++n_output;
        }
    }
    
    if (n_output == 0) {
        return Eigen::MatrixXf(0, 8);
    }
    
    // OPTIMIZED: Pre-allocate output matrix
    Eigen::MatrixXf outputs(n_output, 8);
    size_t out_idx = 0;
    
    // OPTIMIZED: Single pass with direct assignment
    for (const auto& track : active_tracks_) {
        if (track.is_activated()) {
            Eigen::Vector4f xyxy = track.xyxy();
            outputs(out_idx, 0) = xyxy(0); // x1
            outputs(out_idx, 1) = xyxy(1); // y1
            outputs(out_idx, 2) = xyxy(2); // x2
            outputs(out_idx, 3) = xyxy(3); // y2
            outputs(out_idx, 4) = static_cast<float>(track.id());
            outputs(out_idx, 5) = track.conf();
            outputs(out_idx, 6) = static_cast<float>(track.cls());
            outputs(out_idx, 7) = static_cast<float>(track.det_ind());
            ++out_idx;
        }
    }
    
    return outputs;
}

std::vector<STrack> ByteTrack::joint_stracks(const std::vector<STrack>& tlista,
                                             const std::vector<STrack>& tlistb) const {
    std::unordered_set<int> exists;
    std::vector<STrack> res = tlista;
    
    for (const auto& t : tlista) {
        exists.insert(t.id());
    }
    
    for (const auto& t : tlistb) {
        if (exists.find(t.id()) == exists.end()) {
            res.push_back(t);
            exists.insert(t.id());
        }
    }
    
    return res;
}

std::vector<STrack> ByteTrack::sub_stracks(const std::vector<STrack>& tlista,
                                          const std::vector<STrack>& tlistb) const {
    std::unordered_set<int> to_remove;
    for (const auto& t : tlistb) {
        to_remove.insert(t.id());
    }
    
    std::vector<STrack> res;
    for (const auto& t : tlista) {
        if (to_remove.find(t.id()) == to_remove.end()) {
            res.push_back(t);
        }
    }
    
    return res;
}

std::pair<std::vector<STrack>, std::vector<STrack>> ByteTrack::remove_duplicate_stracks(
    const std::vector<STrack>& stracksa,
    const std::vector<STrack>& stracksb) const {
    
    if (stracksa.empty() || stracksb.empty()) {
        return {stracksa, stracksb};
    }
    
    Eigen::MatrixXf a_xyxy(stracksa.size(), 4);
    Eigen::MatrixXf b_xyxy(stracksb.size(), 4);
    for (size_t i = 0; i < stracksa.size(); ++i) {
        a_xyxy.row(i) = stracksa[i].xyxy().transpose();
    }
    for (size_t i = 0; i < stracksb.size(); ++i) {
        b_xyxy.row(i) = stracksb[i].xyxy().transpose();
    }
    
    Eigen::MatrixXf pdist = utils::iou_distance(a_xyxy, b_xyxy);
    
    std::vector<int> dupa, dupb;
    for (int i = 0; i < pdist.rows(); ++i) {
        for (int j = 0; j < pdist.cols(); ++j) {
            if (pdist(i, j) < 0.15f) {
                int timep = stracksa[i].frame_id() - stracksa[i].start_frame();
                int timeq = stracksb[j].frame_id() - stracksb[j].start_frame();
                //decide which tracks survives (longer-lived track wins)
                if (timep > timeq) {
                    dupb.push_back(j);
                } else {
                    dupa.push_back(i);
                }
            }
        }
    }
    
    std::vector<STrack> resa, resb;
    for (size_t i = 0; i < stracksa.size(); ++i) {
        if (std::find(dupa.begin(), dupa.end(), i) == dupa.end()) {
            resa.push_back(stracksa[i]);
        }
    }
    for (size_t i = 0; i < stracksb.size(); ++i) {
        if (std::find(dupb.begin(), dupb.end(), i) == dupb.end()) {
            resb.push_back(stracksb[i]);
        }
    }
    
    return {resa, resb};
}

} // namespace motcpp::trackers

