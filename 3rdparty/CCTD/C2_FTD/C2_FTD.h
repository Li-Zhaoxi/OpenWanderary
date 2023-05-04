#ifndef _C2_FTD_H_
#define _C2_FTD_H_

#include <vector>
#include <opencv2/opencv.hpp>

struct LineINEllipse
{
	int idxEllipse;   //��Ӧ��Բ���±꣬ Ĭ��Ϊ-1
	std::vector<cv::Vec4i> line_data; //
	float lineScore;
	float cross4Direct[4];
};

class C2_FTD
{
public:
	C2_FTD();
	void create(int rows, int cols);
	C2_FTD(int rows, int cols);
	void runC2_FTD(double *_elps, int elpnum, signed int *_linesegs, int linenum, int irows, int icols)
	{
		std::vector<cv::RotatedRect> detEllipses;
		std::vector<cv::Vec4i> detLineSegments;
		for(int i = 0; i < elpnum; i++)
		{
			cv::RotatedRect tmp;
			tmp.center.x = _elps[i * 5 + 0];
			tmp.center.y = _elps[i * 5 + 1];
			tmp.size.width = _elps[i * 5 + 2];
			tmp.size.height = _elps[i * 5 + 3];
			tmp.angle = _elps[i * 5 + 4];
			detEllipses.push_back(tmp);
		}
		for(int i = 0; i < linenum; i++)
		{
			cv::Vec4i tmp;
			tmp.val[0] = _linesegs[i * 4 + 0];
			tmp.val[1] = _linesegs[i * 4 + 1];
			tmp.val[2] = _linesegs[i * 4 + 2];
			tmp.val[3] = _linesegs[i * 4 + 3];
			detLineSegments.push_back(tmp);
		}
		runC2_FTD(detEllipses, detLineSegments, irows, icols);
	}

	void UpdateResults(double *_landelp, double *_landcross)
	{
		_landelp[0] = landEllipse.center.x;
		_landelp[1] = landEllipse.center.y;
		_landelp[2] = landEllipse.size.width;
		_landelp[3] = landEllipse.size.height;
		_landelp[4] = landEllipse.angle;

		_landcross[0] = landCross[0];
		_landcross[1] = landCross[1];
		_landcross[2] = landCross[2];
		_landcross[3] = landCross[3];
		
	}
	void runC2_FTD(std::vector<cv::RotatedRect> &detEllipses, std::vector<cv::Vec4i> &detLineSegments, int rows, int cols);

	int get_insidelinenum() 
	{
		if (!isFind)
			return 0;
		return landLines.size(); 
	}
	void UpdateInsideLines(signed int *_insidelines)
	{
		for(int i = 0; i < landLines.size(); i++)
		{
			_insidelines[i * 4 + 0] = landLines[i][0];
			_insidelines[i * 4 + 1] = landLines[i][1];
			_insidelines[i * 4 + 2] = landLines[i][2];
			_insidelines[i * 4 + 3] = landLines[i][3];
		}
	}
	void drawInsideLines(unsigned char* _imgC, int rows, int cols, int smallscale)
	{
		cv::Mat imgC(rows, cols, CV_8UC3, _imgC);
		drawInsideLines(imgC, smallscale);
		memcpy(_imgC, imgC.data, rows * cols * 3 * sizeof(unsigned char));
	}
	void drawInsideLines(cv::Mat &ImgC, int smallscale = 1);

	void drawC2_FTD(unsigned char* _imgC, int rows, int cols, int smallscale)
	{
		cv::Mat imgC(rows, cols, CV_8UC3, _imgC);
		drawC2_FTD(imgC, smallscale);
		memcpy(_imgC, imgC.data, rows * cols * 3 * sizeof(unsigned char));
	}
	void drawC2_FTD(cv::Mat &ImgC, int smallscale = 1);
	int isdet() { return isFind? 1: 0;}
	//�������ս����ȡ�������
	cv::RotatedRect landEllipse; //Ŀ����Բ
	cv::Vec4f landCross;         //Ŀ��ʮ����Ϣ
	std::vector<cv::Vec4i> landLines;
	bool isFind;                 //�Ƿ���ҵ�

private:
	cv::Mat lineMap; //ֱ�ߵ�ͼ�����ڲ�������Բ�ڵ�ֱ��
	int drows, dcols, irows, icols;

	cv::Vec4i *_detLines_data;
	std::vector<LineINEllipse> line_in_ellipse;
	std::vector<cv::Vec6d> ellipse_parms;


	// ��irows,icols, ����ֱ�߶ε����������Ϣ
	void getLineMap(unsigned short *_lineMap, std::vector< cv::Vec4i > &cdtLines);

	//��ȡ��ÿ����Բ�ڲ���ֱ�߶� Step1 : Extract Candidate Line Segments of Cross
	void getInsideLines(const ushort *_lineMap, const std::vector<cv::RotatedRect> &cdtEllipse, std::vector<LineINEllipse> &LIE);
	 
	// ����Բ��״����תΪһ�㷽��6������
	static void ELPShape2Equation(const std::vector<cv::RotatedRect> &in_elps, std::vector<cv::Vec6d> &out_parms);


	void SlopeClustering(std::vector<LineINEllipse> &LIE, std::vector<cv::RotatedRect> &detElps);
	void SelectFinalCross(std::vector<LineINEllipse> &LIE, std::vector<cv::RotatedRect> &detEllipses);
	double twoL_means(std::vector<double> &input_lines, std::vector<int> &_bestLabels, double &center_1, double &center_2);
private:

	float fitMat[2][6];
private:

	bool isInEllipse(int x, int y, int elp_idx) const
	{
		const double *_parm = ellipse_parms[elp_idx].val;
		double val;

		val = _parm[0] * x*x + 2 * _parm[1] * x*y + _parm[2] * y*y + 2 * _parm[3] * x + 2 * _parm[4] * y + _parm[5];
		if (val > 0)
			return false;
		else
			return true;
	}

	bool CONSTRAINT_DIST(const cv::Vec4i &line4, const cv::RotatedRect &fitelp) const
	{
		double minDist(1), maxDist;
		double vecL[4], dist;

		maxDist = std::fminf(fitelp.size.height, fitelp.size.width) / 4;

		vecL[0] = line4[2] - line4[0], vecL[1] = line4[3] - line4[1];
		vecL[2] = fitelp.center.x - line4[0]; vecL[3] = fitelp.center.y - line4[1];

		dist = abs(vecL[0] * vecL[3] - vecL[1] * vecL[2]) / sqrt(vecL[0] * vecL[0] + vecL[1] * vecL[1]);

		if (dist < maxDist && dist > minDist)
			return true;
		else
			return false;

	}
};


void drawDetectCross(cv::Mat &imgC, cv::RotatedRect &landEllipse, cv::Vec4f &landCross, cv::Scalar elpcolor, cv::Scalar crosscolor, int thickness);
void drawDetectCross(unsigned char *_imgC, int irows, int icols, double *_landEllipse, double *_landCross, unsigned int *_elpcolor, unsigned int *_crosscolor, int thickness);

void marker2Polar(cv::Mat &imgC, cv::RotatedRect &landEllipse, double scale_st, double scale_ed, int radius_ptnum, cv::Mat &polarC);
void marker2Polar(unsigned char *_imgC, int irows, int icols, double *_landEllipse, double scale_st, double scale_ed, int radius_ptnum, unsigned char *_polarC);
#endif