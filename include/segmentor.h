#pragma once
#ifdef DLL_EXPORT
#define MYDLL __declspec(dllexport)
#else
#define MYDLL __declspec(dllimport)
#endif

#include <opencv2/core.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <pcl/2d/edge.h>

#include "common.h"

class Filter;
class Warpper;

namespace ms {
	class /*MYDLL*/ Segmentor {

	public:
		Segmentor(
			const Intrinsic_& color_intrinsic,
			const Intrinsic_& depth_intrinsic,
			const glm::fmat4x4& d2c_extrinsic);
		~Segmentor();
		
	public:
		cv::Mat doSegmentation(
			const cv::Mat & color, const cv::Mat & depth,
			const MorphSnakeParam & ms_param, const InitLevelSetParam& ls_param,
			int downscale, int mask_in_depth_or_color,
			const VisualizationParam& vs_param);

	private:
		void estimateNormals(pcl::PointCloud<pcl::Normal>::Ptr normal, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double radius);
		void detectDepthEdge(cv::Mat& depth_edge, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const pcl::PointCloud<pcl::Normal>::Ptr normal, float low_thresh, float high_thresh);
		cv::Mat morphological_geodesic_active_contour(const cv::Mat & inv_edge_map, const cv::Mat & merged_edge, const cv::Mat & init_ls, double threshold, int iterations, int smoothing, int ballon);

	private:
		Warpper* warpper_ = nullptr;
		Filter* filter_ = nullptr;
		Intrinsic_ depth_intrinsic_;

		int h_size = 0;
		float h_sum = 0;
	};
};