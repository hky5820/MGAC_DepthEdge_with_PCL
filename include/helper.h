#pragma once
#include <vector>

#include <opencv2/core.hpp>

#define PCL
#ifdef PCL
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#endif

namespace cv_helper {
	std::string type2str(int type) {
		std::string r;

		uchar depth = type & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (type >> CV_CN_SHIFT);

		switch (depth) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
		}

		r += "C";
		r += (chans + '0');

		return r;
	}

	cv::Mat	overlay(const cv::Mat &img1, float w1, const cv::Mat &img2, float w2) {
		cv::Mat ol = cv::Mat::zeros(img1.size(), CV_8UC3);
		uchar* ol_data = (uchar*)ol.data;
		uchar* img1_data = (uchar*)img1.data;
		uchar* img2_data = (uchar*)img2.data;

		uchar chans = 1 + (img2.type() >> CV_CN_SHIFT);

		int width = ol.cols;
		int height = ol.rows;

		if (chans == 1) {
			for (int r = 0; r < height; r++) {
				for (int c = 0; c < width; c++) {
					ol_data[(r*width + c) * 3 + 0] = (int)(img1_data[(r*width + c) * 3 + 0] * w1 + img2_data[r*width + c] * w2);
					ol_data[(r*width + c) * 3 + 1] = (int)(img1_data[(r*width + c) * 3 + 1] * w1 + img2_data[r*width + c] * w2);
					ol_data[(r*width + c) * 3 + 2] = (int)(img1_data[(r*width + c) * 3 + 2] * w1 + img2_data[r*width + c] * w2);
				}
			}
		}
		if (chans == 3) {
			for (int r = 0; r < height; r++) {
				for (int c = 0; c < width; c++) {
					ol_data[(r*width + c) * 3 + 0] = (int)(img1_data[(r*width + c) * 3 + 0] * w1 + img2_data[(r*width + c) * 3 + 0] * w2);
					ol_data[(r*width + c) * 3 + 1] = (int)(img1_data[(r*width + c) * 3 + 1] * w1 + img2_data[(r*width + c) * 3 + 1] * w2);
					ol_data[(r*width + c) * 3 + 2] = (int)(img1_data[(r*width + c) * 3 + 2] * w1 + img2_data[(r*width + c) * 3 + 2] * w2);
				}
			}
		}
		return ol;
	}

	cv::Mat ORimages(const cv::Mat& img1, const cv::Mat& img2) {
		int rows = img1.rows;
		int cols = img2.cols;
		cv::Mat output = cv::Mat::zeros(img1.size(), CV_8UC1);

		uchar* img1_data = (uchar*)img1.data;
		uchar* img2_data = (uchar*)img2.data;
		uchar* output_data = (uchar*)output.data;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (img1_data[r*cols + c] || img2_data[r*cols + c]) {
					output_data[r*cols + c] = 255;
				}
			}
		}

		return output;
	}
}

namespace pc_helper {

	// Point Cloud를 OpenCV 3Channel에 저장
	void depthToPointcloud_Mat(const cv::Mat& depth_image, cv::Mat& pointcloud_xyz, float fx, float fy, float cx, float cy, int downscale) {
		int rows = depth_image.rows;
		int cols = depth_image.cols;
		
		unsigned short* d_img_data = (unsigned short*)depth_image.data;
		float* pc_xyz_data = (float*)pointcloud_xyz.data;
#pragma omp parallel for
		for (int v = 0; v < rows; ++v) {
			for (int u = 0; u < cols; ++u) {
				float Z = d_img_data[v * cols + u];

				float x, y, z;
				z = Z;
				x = (u - cx / downscale) * Z / ( fx / downscale);
				y = (v - cy / downscale) * Z / ( fy / downscale);

				if (z == 0.0) 
					x = y = z =  NAN;
				
				pc_xyz_data[(v * cols + u) * 3 + 0] = x;
				pc_xyz_data[(v * cols + u) * 3 + 1] = y;
				pc_xyz_data[(v * cols + u) * 3 + 2] = z;
			}
		}
	}

	std::vector<float> depthToPointcloud_vec(cv::Mat& depth_image, float fx, float fy, float cx, float cy) {
		std::vector<float> pointcloud_xyz;
#pragma omp parallel for
		for (int v = 0; v < depth_image.rows; ++v) {
			for (int u = 0; u < depth_image.cols; ++u) {
				float Z = depth_image.at<unsigned short>(v, u);

				float x, y, z;
				z = Z; // 카메라의 -z-axis 위에 있음
				x = (u - cx) * Z / fx;
				y = (v - cy) * Z / fy;

				if (x == 0.0 && y == 0.0 && z == 0.0)
					continue;

				pointcloud_xyz.emplace_back(x);
				pointcloud_xyz.emplace_back(y);
				pointcloud_xyz.emplace_back(z);
			}
		}
		return pointcloud_xyz;
	}

	cv::Mat calcNormalFromDepth(const cv::Mat& depth_image) {
		int d_height = depth_image.rows;
		int d_width = depth_image.cols;

		cv::Mat norFloat(depth_image.size(), CV_16SC3);
#pragma omp parallel for
		for (int x = 1; x < d_width; x++) {
			for (int y = 1; y < d_height; y++) {

				cv::Vec3f top(x, y - 1, depth_image.at<unsigned short>(y - 1, x));
				cv::Vec3f left(x - 1, y, depth_image.at<unsigned short>(y, x - 1));
				cv::Vec3f current(x, y, depth_image.at<unsigned short>(y, x));
				cv::Vec3f d = (current - top).cross(current - left);
				cv::Vec3f n = cv::normalize(d);

				short norm_x = (short)(n[0] * (float)SHRT_MAX);
				short norm_y = (short)(n[1] * (float)SHRT_MAX);
				short norm_z = (short)(n[2] * (float)SHRT_MAX);

				norFloat.at<cv::Vec3s>(y, x)[0] = norm_x;
				norFloat.at<cv::Vec3s>(y, x)[1] = norm_y;
				norFloat.at<cv::Vec3s>(y, x)[2] = norm_z;
			}
		}
		return norFloat;
	}

#ifdef  PCL
	void savePointCloutWithMask(const cv::Mat& depth_image, const cv::Mat& mask, float fx, float fy, float cx, float cy) {
		if (!depth_image.data) {
			std::cerr << "No depth data!!!" << std::endl;
			exit(EXIT_FAILURE);
		}
		int rows = depth_image.rows;
		int cols = depth_image.cols;
		unsigned short* d_img_data = (unsigned short*)depth_image.data;
		uchar* mask_data = (uchar*)mask.data;

		pcl::PointCloud<pcl::PointXYZ> pointcloud_xyz_;
		pointcloud_xyz_.height = depth_image.rows;
		pointcloud_xyz_.width = depth_image.cols;
		pointcloud_xyz_.resize(pointcloud_xyz_.height * pointcloud_xyz_.width);

#pragma omp parallel for
		for (int v = 0; v < depth_image.rows; ++v) {
			for (int u = 0; u < depth_image.cols; ++u) {
				float Z = d_img_data[v * cols + u];

				float x, y, z;
				z = Z;
				x = (u - cx) * Z / fx;
				y = (v - cy) * Z / fy;

				if (x == 0.0 && y == 0.0 && z == 0.0)
					continue;

				if (mask_data[v*cols + u]) {
					pcl::PointXYZ xyz;
					xyz.x = x;
					xyz.y = y;
					xyz.z = z;
					pointcloud_xyz_.at(u, v) = xyz;
				}
			}
		}

		pcl::io::savePLYFile("pc.ply", pointcloud_xyz_);
}

	void depthToPointcloud_pcl(const cv::Mat& color_image, const cv::Mat& depth_image, float fx, float fy, float cx, float cy, int downscale, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_xyz) {
		if (!depth_image.data) {
			std::cerr << "No depth data!!!" << std::endl;
			exit(EXIT_FAILURE);
		}

		(*pointcloud_xyz).height = depth_image.rows;
		(*pointcloud_xyz).width = depth_image.cols;
		(*pointcloud_xyz).resize((*pointcloud_xyz).height * (*pointcloud_xyz).width);
		(*pointcloud_xyz).is_dense = false;

#pragma omp parallel for
		for (int v = 0; v < depth_image.rows; ++v) {
			for (int u = 0; u < depth_image.cols; ++u) {
				float Z = depth_image.at<unsigned short>(v, u);

				float x, y, z;
				z = Z;
				x = (u - (cx / downscale)) * Z / (fx / downscale);
				y = (v - (cy / downscale)) * Z / (fy / downscale);

				if (z == 0.0) 
					x = y = z = NAN;
				

				pcl::PointXYZRGB xyzrgb;
				xyzrgb.x = x / 1000;
				xyzrgb.y = y / 1000;
				xyzrgb.z = z / 1000;
				xyzrgb.r = color_image.at<cv::Vec3b>(v, u)[2];
				xyzrgb.g = color_image.at<cv::Vec3b>(v, u)[1];
				xyzrgb.b = color_image.at<cv::Vec3b>(v, u)[0];

				(*pointcloud_xyz).at(u, v) = xyzrgb;
			}
		}
	}
#endif //  PCL
}