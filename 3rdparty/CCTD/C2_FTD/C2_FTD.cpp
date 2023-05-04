#include "C2_FTD/C2_FTD.h"

C2_FTD::C2_FTD()
{
	isFind = false;
	_detLines_data = NULL;


	icols = irows = drows = dcols = -1;

}

void C2_FTD::create(int rows, int cols)
{
	drows = rows, dcols = cols;
	lineMap = cv::Mat::zeros(rows, cols, CV_16UC1);
	isFind = false;
	_detLines_data = NULL;
}
C2_FTD::C2_FTD(int rows, int cols)
{
	create(rows, cols);
}

void C2_FTD::runC2_FTD(std::vector<cv::RotatedRect> &detEllipses, std::vector<cv::Vec4i> &detLineSegments, int rows, int cols)
{
//  �����ʼ��
	isFind = false;
	landEllipse.center.x = landEllipse.center.y = 0;
	landEllipse.size.width = landEllipse.size.height = 1;

	line_in_ellipse.clear();
	if (detEllipses.size() == 0)
		return;

	irows = rows, icols = cols;
	unsigned short *_lineMap = (unsigned short*)lineMap.data;
	_detLines_data = detLineSegments.data();

	getLineMap(_lineMap, detLineSegments);    // ����ֱ�ߵ�ͼ

	ELPShape2Equation(detEllipses, ellipse_parms);
	getInsideLines(_lineMap, detEllipses, line_in_ellipse);

	SlopeClustering(line_in_ellipse, detEllipses);
	SelectFinalCross(line_in_ellipse, detEllipses);

}


void C2_FTD::getLineMap(unsigned short *_lineMap, std::vector< cv::Vec4i > &cdtLines)
{
	memset((void*)_lineMap, 0, sizeof(unsigned short)*drows*dcols);

	const int lineNum = (int)cdtLines.size();
	for (int i = 0; i < lineNum; i++)
		_lineMap[cdtLines[i][0] * icols + cdtLines[i][1]] = i + 1;
}

void C2_FTD::getInsideLines(const ushort *_lineMap, const std::vector<cv::RotatedRect> &cdtEllipse, std::vector<LineINEllipse> &LIE)
{
	cv::Point loc[4];
	unsigned short val;

	const int elpNum = (int)cdtEllipse.size();

	LIE.resize(elpNum);
	for (int i = 0; i < elpNum; i++)
	{
		// Get search range.
		loc[0].x = int(cdtEllipse[i].center.x - cdtEllipse[i].size.width / 2 + 0.5), loc[0].y = int(cdtEllipse[i].center.y - cdtEllipse[i].size.width / 2 + 0.5);
		loc[1].x = int(cdtEllipse[i].center.x - cdtEllipse[i].size.width / 2 + 0.5), loc[1].y = int(cdtEllipse[i].center.y + cdtEllipse[i].size.width / 2 + 0.5);
		loc[2].x = int(cdtEllipse[i].center.x + cdtEllipse[i].size.width / 2 + 0.5), loc[2].y = int(cdtEllipse[i].center.y + cdtEllipse[i].size.width / 2 + 0.5);
		loc[3].x = int(cdtEllipse[i].center.x + cdtEllipse[i].size.width / 2 + 0.5), loc[3].y = int(cdtEllipse[i].center.y - cdtEllipse[i].size.width / 2 + 0.5);
		for (int j = 0; j < 4; j++)
		{
			if (loc[j].x < 0) loc[j].x = 0;
			if (loc[j].x >= irows) loc[j].x = irows - 1;
			if (loc[j].y < 0) loc[j].y = 0;
			if (loc[j].y >= icols) loc[j].y = icols - 1;
		}

		// Find Lines;
		LIE[i].idxEllipse = i;//������Բ�Ǳ�
		LIE[i].line_data.clear(); // ���ֱ�߶�����

		for (int iMap = loc[0].x; iMap <= loc[3].x; iMap++)
		{
			for (int jMap = loc[0].y; jMap <= loc[1].y; jMap++)
			{
				val = _lineMap[iMap*icols + jMap]; //���ֱ�߶ζ�Ӧ��index
				if (val == 0) //�˴�û��ֱ��
					continue;
				if (!isInEllipse(iMap, jMap, i))//ֱ�ߵ�һ���˵㲻����Բ�ڲ�
					continue;
				val--; //���ֱ�߶�idx
				if (!isInEllipse(_detLines_data[val][2], _detLines_data[val][3], i)) // ֱ�ߵ���һ���˵㲻����Բ�ڲ�
					continue;
				if (!CONSTRAINT_DIST(_detLines_data[val], cdtEllipse[i])) //������ֱ���������Լ��
					continue;
				LIE[i].line_data.push_back(_detLines_data[val]);
			}
		}
	}
}


void C2_FTD::SlopeClustering(std::vector<LineINEllipse> &LIE, std::vector<cv::RotatedRect> &detElps)
{
	const int all_num = (int)LIE.size(); //�洢������������Բ����

	double cluster_err, angleCluse[2], dx, dy;
	std::vector<double> slopes;
	std::vector<int> best_labels;
	

	for (int i = 0; i < all_num; i++)
	{
		const int line_num = (int)LIE[i].line_data.size();
		// std::cout << "line_num: " << line_num << std::endl;
		if (line_num <= 4) //С��4��ֱ��ȱʧ���أ��ӵ�
		{
			LIE[i].lineScore = 100;
			continue;
		}

		// ֱ�߶� б�� ����
		slopes.resize(line_num);
		cv::Vec4i *_line_data = LIE[i].line_data.data();
		for (int j = 0; j < line_num; j++)
		{
			dx = _line_data[j][2] - _line_data[j][0];
			dy = _line_data[j][3] - _line_data[j][1];
			if (dx < 0) { dx = -dx, dy = -dy; }
			slopes[j] = atan2(dy, dx);
		}

		cluster_err = twoL_means(slopes, best_labels, angleCluse[0], angleCluse[1]) / CV_PI * 180;
		cluster_err /= line_num;
		// ������������ֱ�߶εĽ���
		for (int j = 0; j < 6; j++)
			fitMat[0][j] = 0, fitMat[1][j] = 0;
		LIE[i].cross4Direct[0] = float(cos(angleCluse[0])), LIE[i].cross4Direct[1] = float(sin(angleCluse[0]));
		LIE[i].cross4Direct[2] = float(cos(angleCluse[1])), LIE[i].cross4Direct[3] = float(sin(angleCluse[1]));

		LIE[i].lineScore = (float)cluster_err;

		// std::cout << "cluster_err: " << cluster_err << std::endl;
		
	}

}


void C2_FTD::ELPShape2Equation(const std::vector<cv::RotatedRect> &in_elps, std::vector<cv::Vec6d> &out_parms)
{
	const int elp_num = (int)in_elps.size();

	double xc, yc, a, b, theta, sqr_a, sqr_b;
	double cos_theta, sin_theta ,sqr_cos_theta, sqr_sin_theta;
	double *parm_temp(NULL);

	out_parms.resize(elp_num);
	for (int i = 0; i < elp_num; i++)
	{
		xc = in_elps[i].center.x;
		yc = in_elps[i].center.y;
		a = in_elps[i].size.width;
		b = in_elps[i].size.height;
		theta = in_elps[i].angle / 180 * CV_PI;

		cos_theta = cos(theta), sin_theta = sin(theta);
		sqr_cos_theta = cos_theta*cos_theta, sqr_sin_theta = sin_theta*sin_theta;

		sqr_a = a*a, sqr_b = b*b;

		parm_temp = out_parms[i].val;
		parm_temp[0] = sqr_cos_theta / sqr_a + sqr_sin_theta / sqr_b;
		parm_temp[1] = -(sin(2 * theta)*(sqr_a - sqr_b)) / (2 * sqr_a*sqr_b);
		parm_temp[2] = sqr_cos_theta / sqr_b + sqr_sin_theta / sqr_a;
		parm_temp[3] = (-sqr_a*xc*sqr_sin_theta + (sqr_a*yc*sin(2 * theta)) / 2) / (sqr_a*sqr_b) - (xc*sqr_cos_theta + (yc*sin(2 * theta)) / 2) / sqr_a;
		parm_temp[4] = (-sqr_a*yc*sqr_cos_theta + (sqr_a*xc*sin(2 * theta)) / 2) / (sqr_a*sqr_b) - (yc*sqr_sin_theta + (xc*sin(2 * theta)) / 2) / sqr_a;
		parm_temp[5] = (xc*cos_theta + yc*sin_theta)*(xc*cos_theta + yc*sin_theta) / sqr_a + (yc*cos_theta - xc*sin_theta)*(yc*cos_theta - xc*sin_theta) / sqr_b - 1;

	}
}


void C2_FTD::SelectFinalCross(std::vector<LineINEllipse> &LIE, std::vector<cv::RotatedRect> &detEllipses)
{
	const int LIE_num = (int)LIE.size();
	int idx_min = 0;
	float val_min = LIE[0].lineScore;
	for (int i = 1; i < LIE_num; i++)
	{
		if (LIE[i].lineScore < val_min)
			val_min = LIE[i].lineScore, idx_min = i;
	}

	if (val_min < 50)
	{
		isFind = true;
		landEllipse = detEllipses[idx_min];
		landCross[0] = LIE[idx_min].cross4Direct[0];
		landCross[1] = LIE[idx_min].cross4Direct[1];
		landCross[2] = LIE[idx_min].cross4Direct[2];
		landCross[3] = LIE[idx_min].cross4Direct[3];

		landLines = LIE[idx_min].line_data;
	}
}


void marker2Polar(cv::Mat &imgC, cv::RotatedRect &landEllipse, double scale_st, double scale_ed, int radius_ptnum, cv::Mat &polarC)
{
	if (imgC.channels() == 1)
		cvtColor(imgC, imgC, cv::COLOR_GRAY2BGR);

	const int irows = imgC.rows, icols = imgC.cols;
	const int thetanum = 360;
	std::vector<cv::Mat> vldBaseData(thetanum);
	double h = CV_PI * 2 / thetanum;
	for(int i = 0; i < thetanum; i++)
	{
		cv::Mat tmp(2, 1, CV_64FC1);
		tmp.at<double>(0) = cos(i * h);
		tmp.at<double>(1) = sin(i * h);
		tmp.copyTo(vldBaseData[i]);
	}

	polarC.create(thetanum, radius_ptnum, CV_8UC3);
	double step = (scale_ed - scale_st) / radius_ptnum;
	double R = landEllipse.size.height / 2;
	double r = landEllipse.size.width / 2;
	double angleRot = -landEllipse.angle / 180.0 * CV_PI;

	cv::Mat matrot(2, 2, CV_64FC1);
	matrot.at<double>(0, 0) = R * cos(angleRot);
	matrot.at<double>(0, 1) = -r * sin(angleRot);
	matrot.at<double>(1, 0) = R * sin(angleRot);
	matrot.at<double>(1, 1) = r *cos(angleRot);
	

	cv::Mat cen(2, 1, CV_64FC1);
	cen.at<double>(0) = landEllipse.center.y;
	cen.at<double>(1) = landEllipse.center.x;

	for (int theta = 0; theta < thetanum; theta++)
	{
		cv::Mat tmp = matrot * vldBaseData[theta];
		for (int idxradius = 0; idxradius < radius_ptnum; idxradius++)
		{
			double scale = scale_st + step * idxradius;
			cv::Mat pt = scale * tmp + cen;

			int idxu = int(pt.at<double>(0) + 0.5);
			int idxv = int(pt.at<double>(1) + 0.5);
			if (idxu < 0 || idxv < 0 || idxu >= irows || idxv >= icols)
			{
				polarC.at<cv::Vec3b>(theta, idxradius)[0] = 0;
				polarC.at<cv::Vec3b>(theta, idxradius)[1] = 255;
				polarC.at<cv::Vec3b>(theta, idxradius)[2] = 0;
			}
			else
			{
				polarC.at<cv::Vec3b>(theta, idxradius)[0] = polarC.at<cv::Vec3b>(idxu, idxv)[0];
				polarC.at<cv::Vec3b>(theta, idxradius)[1] = polarC.at<cv::Vec3b>(idxu, idxv)[1];
				polarC.at<cv::Vec3b>(theta, idxradius)[2] = polarC.at<cv::Vec3b>(idxu, idxv)[2];
			}
		}
	}
}


void marker2Polar(unsigned char *_imgC, int irows, int icols, double *_landEllipse, double scale_st, double scale_ed, int radius_ptnum, unsigned char *_polarC)
{
	cv::Mat imgC(irows, icols, CV_8UC3, _imgC), polarC;
	cv::RotatedRect landEllipse;
	landEllipse.center.x = _landEllipse[0];
	landEllipse.center.y = _landEllipse[1];
	landEllipse.size.width = _landEllipse[2];
	landEllipse.size.height = _landEllipse[3];
	landEllipse.angle = _landEllipse[4];

	marker2Polar(imgC, landEllipse, scale_st, scale_ed, radius_ptnum, polarC);
	memcpy(_polarC, polarC.data, sizeof(unsigned char) * 3 * radius_ptnum + 360);
}