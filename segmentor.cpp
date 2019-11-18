#include "segmentor.h"

#include <chrono>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "helper.h"
#include "warpper.h"
#include "filter.h"

#include <pcl/features/normal_3d.h>
#include <pcl/2d/convolution.h>

typedef std::chrono::high_resolution_clock::time_point c_time;

using namespace ms;

Segmentor::Segmentor(const Intrinsic_ & color_intrinsic, const Intrinsic_ & depth_intrinsic, const glm::fmat4x4 & d2c_extrinsic){
	depth_intrinsic_ = depth_intrinsic;
	warpper_ = new Warpper(color_intrinsic, depth_intrinsic, d2c_extrinsic);
	filter_ = new Filter();
}

Segmentor::~Segmentor() {
	if (warpper_)
		delete warpper_;

	if (filter_)
		delete filter_;
}

cv::Mat Segmentor::doSegmentation(
	const cv::Mat& color, 
	const cv::Mat& depth, 
	const MorphSnakeParam & ms_param, 
	const InitLevelSetParam& ls_param,
	int downscale, 
	int mask_in_depth_or_color,
	const VisualizationParam& vs_param) {

	cv::Mat resized_color, resized_depth;
	cv::resize(color, resized_color, cv::Size(color.cols / downscale, color.rows / downscale));
	cv::resize(depth, resized_depth, cv::Size(depth.cols / downscale, depth.rows / downscale));

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pc_helper::depthToPointcloud_pcl(resized_color, resized_depth, depth_intrinsic_.fx, depth_intrinsic_.fy, depth_intrinsic_.ppx, depth_intrinsic_.ppy, downscale, cloud);

	cv::Mat warpped_color = cv::Mat::zeros(resized_depth.rows, resized_depth.cols, CV_8UC3);
	warpper_->setHomography(resized_color, cloud, downscale);
	warpper_->warpRGB_CS2DS(resized_color, warpped_color, INTERPOLATION::bilinear);
	if (vs_param.warpping_on) {
		cv::Mat output;
		cv::resize(warpped_color, output, cv::Size(warpped_color.cols * downscale, warpped_color.rows * downscale));
		return output;
	}
	
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
	estimateNormals(normal, cloud, 0.005);

	cv::Mat d_edge = cv::Mat::zeros(resized_depth.size(), CV_8UC1);
	detectDepthEdge(d_edge, cloud, normal, 0.5f, 0.8f);
	cv::imshow("d_edge", d_edge);

	// Resized Warpped RGB to Gray (16 bit)
	cv::Mat gray = filter_->RGB2GRAY(warpped_color, ms_param.channel);
	
	// Making Inverse Edge Map
	int window_size = 3;
	cv::Mat inv_edge_map = filter_->inverse_edge_map(gray, ms_param.alpha, ms_param.sigma, window_size);
	if (vs_param.inv_edge_on) 
		cv::imshow("Inv_Edge", inv_edge_map);

	// Make Init Level Set (Circle)
	cv::Mat init_ls = filter_->make_init_ls({ inv_edge_map.rows , inv_edge_map.cols }, { ls_param.center_row / downscale, ls_param.center_col / downscale }, ls_param.radius);

	// Morphological Snake
	cv::Mat mask = morphological_geodesic_active_contour(inv_edge_map, d_edge, init_ls, ms_param.threshold, ms_param.iteration,  ms_param.smoothing, ms_param.ballon);

	if (mask_in_depth_or_color == MASK_AT::COLOR) {
		cv::Mat warpped_mask  = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
		warpper_->warpGray_DS2CS(mask, warpped_mask);

		cv::Mat output;
		cv::resize(warpped_mask, output, cv::Size(warpped_mask.cols * downscale, warpped_mask.rows * downscale));

		return output;
	}

	// Return Mask at Depth Space
	cv::Mat output;
	cv::resize(mask, output, cv::Size(mask.cols * downscale, mask.rows * downscale));
	return output;
}

void ms::Segmentor::estimateNormals(pcl::PointCloud<pcl::Normal>::Ptr normal, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double radius){
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.setRadiusSearch(radius);
	ne.compute(*normal);
}

void ms::Segmentor::detectDepthEdge(cv::Mat & depth_edge, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const pcl::PointCloud<pcl::Normal>::Ptr normal, float low_thresh, float high_thresh){
	/* High curvature edge */
// nx ny
	pcl::PointCloud<pcl::PointXYZI> nx, ny;
	nx.width = normal->width;
	nx.height = normal->height;
	nx.resize(normal->height*normal->width);

	ny.width = normal->width;
	ny.height = normal->height;
	ny.resize(normal->height*normal->width);

	for (std::uint32_t row = 0; row < normal->height; row++) {
		for (std::uint32_t col = 0; col < normal->width; col++) {
			nx(col, row).intensity = normal->points[row*normal->width + col].normal_x;
			ny(col, row).intensity = normal->points[row*normal->width + col].normal_y;
		}
	}

	// canny
	pcl::PointCloud<pcl::PointXYZIEdge> img_edge;
	pcl::Edge<pcl::PointXYZI, pcl::PointXYZIEdge> edge;
	edge.setHysteresisThresholdLow(low_thresh);
	edge.setHysteresisThresholdHigh(high_thresh);
	edge.canny(nx, ny, img_edge);

	// edge
	for (std::uint32_t row = 0; row < normal->height; row++) {
		for (std::uint32_t col = 0; col < normal->width; col++) {
			if (img_edge(col, row).magnitude == 255.f) {
				depth_edge.at<unsigned char>(row, col) = 255;
			}
		}
	}

	/* Boudary edge */
	struct Neighbor {
		Neighbor(int dx, int dy, int didx)
			: d_x(dx)
			, d_y(dy)
			, d_index(didx)
		{}

		int d_x;
		int d_y;
		int d_index; // = dy * width + dx: pre-calculated
	};

	int max_search_neighbors_(50);
	float th_depth_discon_(0.02f);

	const int num_of_ngbr = 8;
	Neighbor directions[num_of_ngbr] = { Neighbor(-1, 0, -1),
	  Neighbor(-1, -1, -cloud.get()->width - 1),
	  Neighbor(0, -1, -cloud.get()->width),
	  Neighbor(1, -1, -cloud.get()->width + 1),
	  Neighbor(1,  0,                 1),
	  Neighbor(1,  1,  cloud.get()->width + 1),
	  Neighbor(0,  1,  cloud.get()->width),
	  Neighbor(-1,  1,  cloud.get()->width - 1) };

	for (int r = 1; r < cloud.get()->height - 1; ++r) {
		for (int c = 1; c < cloud.get()->width - 1; ++c) {
			int curr_idx = r * int(cloud.get()->width) + c;
			if (!std::isfinite(cloud.get()->points[curr_idx].z))
				continue;

			float curr_depth = std::abs(cloud.get()->points[curr_idx].z);

			// Calculate depth distances between current point and neighboring points
			std::vector<float> nghr_dist;
			nghr_dist.resize(8);

			bool found_invalid_neighbor = false;
			for (int d_idx = 0; d_idx < num_of_ngbr; d_idx++)
			{
				int nghr_idx = curr_idx + directions[d_idx].d_index;
				assert(nghr_idx >= 0 && nghr_idx < cloud.get()->points.size());
				if (!std::isfinite(cloud.get()->points[nghr_idx].z))
				{
					found_invalid_neighbor = true;
					break;
				}
				nghr_dist[d_idx] = curr_depth - std::abs(cloud.get()->points[nghr_idx].z);
			}

			if (!found_invalid_neighbor) {
				// Every neighboring points are valid
				std::vector<float>::iterator min_itr = std::min_element(nghr_dist.begin(), nghr_dist.end());
				std::vector<float>::iterator max_itr = std::max_element(nghr_dist.begin(), nghr_dist.end());
				float nghr_dist_min = *min_itr;
				float nghr_dist_max = *max_itr;
				float dist_dominant = std::abs(nghr_dist_min) > std::abs(nghr_dist_max) ? nghr_dist_min : nghr_dist_max;
				if (std::abs(dist_dominant) > th_depth_discon_*std::abs(curr_depth)) {
					// Found a depth discontinuity
					if (dist_dominant > 0.f) {
						//if (detecting_edge_types_ & EDGELABEL_OCCLUDED)
						//   labels[curr_idx].label |= EDGELABEL_OCCLUDED;
						depth_edge.at<unsigned char>(r, c) = 255;
					}
					else {
						//if (detecting_edge_types_ & EDGELABEL_OCCLUDING)
						//   labels[curr_idx].label |= EDGELABEL_OCCLUDING;
						depth_edge.at<unsigned char>(r, c) = 255;
					}
				}
			}
			else {
				// Some neighboring points are not valid (nan points)
				// Search for corresponding point across invalid points
				// Search direction is determined by nan point locations with respect to current point
				int dx = 0;
				int dy = 0;
				int num_of_invalid_pt = 0;
				for (const auto &direction : directions) {
					int nghr_idx = curr_idx + direction.d_index;
					assert(nghr_idx >= 0 && nghr_idx < cloud.get()->points.size());
					if (!std::isfinite(cloud.get()->points[nghr_idx].z)) {
						dx += direction.d_x;
						dy += direction.d_y;
						num_of_invalid_pt++;
					}
				}

				// Search directions
				assert(num_of_invalid_pt > 0);
				float f_dx = static_cast<float> (dx) / static_cast<float> (num_of_invalid_pt);
				float f_dy = static_cast<float> (dy) / static_cast<float> (num_of_invalid_pt);

				// Search for corresponding point across invalid points
				float corr_depth = std::numeric_limits<float>::quiet_NaN();
				for (int s_idx = 1; s_idx < max_search_neighbors_; s_idx++) {
					int s_row = r + static_cast<int> (std::floor(f_dy*static_cast<float> (s_idx)));
					int s_col = c + static_cast<int> (std::floor(f_dx*static_cast<float> (s_idx)));

					if (s_row < 0 || s_row >= int(cloud.get()->height) || s_col < 0 || s_col >= int(cloud.get()->width))
						break;

					if (std::isfinite(cloud.get()->points[s_row*int(cloud.get()->width) + s_col].z)) {
						corr_depth = std::abs(cloud.get()->points[s_row*int(cloud.get()->width) + s_col].z);
						break;
					}
				}

				if (!std::isnan(corr_depth)) {
					// Found a corresponding point
					float dist = curr_depth - corr_depth;
					if (std::abs(dist) > th_depth_discon_*std::abs(curr_depth)) {
						// Found a depth discontinuity
						if (dist > 0.f) {
							//      if (detecting_edge_types_ & EDGELABEL_OCCLUDED)
							//         labels[curr_idx].label |= EDGELABEL_OCCLUDED;
							depth_edge.at<unsigned char>(r, c) = 255;
						}
						else {
							//      if (detecting_edge_types_ & EDGELABEL_OCCLUDING)
							//         labels[curr_idx].label |= EDGELABEL_OCCLUDING;
							depth_edge.at<unsigned char>(r, c) = 255;
						}
					}
				}
				else {
					//// Not found a corresponding point, just nan boundary edge
					//if (detecting_edge_types_ & EDGELABEL_NAN_BOUNDARY)
					   //labels[curr_idx].label |= EDGELABEL_NAN_BOUNDARY;
				}
			}
		}
	}
}

cv::Mat Segmentor::morphological_geodesic_active_contour(
	const cv::Mat & inv_edge_map,
	const cv::Mat& merged_edge,
	const cv::Mat & init_ls,
	double threshold,
	int iterations,
	int smoothing,
	int ballon) {

	double* inv_edge_map_data = (double*)inv_edge_map.data;
	uchar* me_data = (uchar*)merged_edge.data;

	int rows = inv_edge_map.rows;
	int	cols = inv_edge_map.cols;

	cv::Mat threshold_mask_balloon = cv::Mat::zeros(rows, cols, CV_8UC1);
	uchar* tmb_data = (uchar*)threshold_mask_balloon.data;
	int abs_ballon = ballon < 0 ? -ballon : ballon; // abs(ballon)
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if (inv_edge_map_data[r * cols + c] > (threshold / (double)abs_ballon))
				tmb_data[r * cols + c] = 255;
		}
	}

	cv::Mat gx = cv::Mat::zeros(rows, cols, CV_64FC1);
	cv::Mat	gy = cv::Mat::zeros(rows, cols, CV_64FC1);
	filter_->gradient(inv_edge_map, gx, gy);
	double* gx_data = (double*)gx.data;
	double* gy_data = (double*)gy.data;

	cv::Mat dgx = cv::Mat::zeros(rows, cols, CV_64FC1);
	cv::Mat	dgy = cv::Mat::zeros(rows, cols, CV_64FC1);
	std::vector<cv::Mat> temps = filter_->get_structures();

	cv::Mat u = init_ls;
	cv::Mat du;
	cv::Mat aux;
	cv::Mat structure = cv::Mat::ones(3, 3, CV_8UC1);
	for (int c_itr = 0; c_itr < iterations; c_itr++) {
		if (ballon > 0)
			cv::dilate(u, aux, structure);
		else if (ballon < 0)
			cv::erode(u, aux, structure);

		uchar* u_data = (uchar*)u.data;
		uchar* aux_data = (uchar*)aux.data;
		if (ballon != 0) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (tmb_data[r * cols + c])
						u_data[r * cols + c] = aux_data[r * cols + c];
				}
			}
		}

		u.convertTo(du, CV_64FC1, 1. / 255.);
		filter_->gradient(du, dgx, dgy);
		double* dgx_data = (double*)dgx.data;
		double* dgy_data = (double*)dgy.data;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				double val =
					gx_data[r*cols + c] * dgx_data[r*cols + c]
					+ gy_data[r*cols + c] * dgy_data[r*cols + c];
				if (val > 0)
					u_data[r * cols + c] = 255;
				else if (val < 0)
					u_data[r * cols + c] = 0;
			}
		}
		for (int i = 0; i < smoothing; i++)
			u = filter_->smoothing(u, temps);
	}
	return u;
}