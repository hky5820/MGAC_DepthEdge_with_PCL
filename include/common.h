#pragma once
#include <opencv2/core.hpp>

#include <glm/glm.hpp>

namespace ms {
	enum INTERPOLATION {
		bilinear = 0
	};


	enum CHANNEL {
		RED = 0,
		GREEN,
		BLUE,
		GRAY
	};

	enum MASK_AT {
		COLOR = 0,
		DEPTH
	};

	struct MorphSnakeParam {
		MorphSnakeParam() {};
		MorphSnakeParam(
			double alpha_,
			double sigma_,
			int channel_,
			int iteration_,
			double threshold_,
			int smoothing_,
			int ballon_) :
			alpha(alpha_),
			sigma(sigma_),
			channel(channel_),
			iteration(iteration_),
			threshold(threshold_),
			smoothing(smoothing_),
			ballon(ballon_) {}
		double alpha;
		double sigma;  // Gaussian에 사용되는 sigma
		int channel;   // RGB에서 Gray로 변경시, 사용될 채널 종류(first,second,third) 혹은 OpenCV cvtColor 함수 ( GRAY )
		int iteration;
		double threshold;
		int smoothing; // smoothing 횟수
		int ballon;    // 1 or -1 ( 팽창 or 축소 )
	};

	struct InitLevelSetParam {
		InitLevelSetParam() {};
		InitLevelSetParam(
			int center_row_,
			int center_col_,
			int radius_) :
			center_row(center_row_),
			center_col(center_col_),
			radius(radius_) {}

		int center_row;
		int center_col;
		int radius;
	};

	struct VisualizationParam {
		VisualizationParam(bool inv_edge_on_, bool depth_edge_on_, bool warpping_on_) :
			inv_edge_on(inv_edge_on_),
			depth_edge_on(depth_edge_on_),
			warpping_on(warpping_on_) {}
		bool inv_edge_on;
		bool depth_edge_on;
		bool warpping_on;
	};

	struct Intrinsic_ {
		Intrinsic_() {};
		Intrinsic_(
			float fx_,
			float fy_,
			float ppx_,
			float ppy_,
			int width_,
			int height_) :
			fx(fx_),
			fy(fy_),
			ppx(ppx_),
			ppy(ppy_),
			width(width_),
			height(height_) {}
		float fx;
		float fy;
		float ppx;
		float ppy;
		int width;
		int height;
	};
};