#include "C2_FTD/C2_FTD.h"

void C2_FTD::drawInsideLines(cv::Mat &ImgC, int smallscale)
{
	if (ImgC.channels() == 1)
		cv::cvtColor(ImgC, ImgC, cv::COLOR_GRAY2BGR);
	
	for (int i = 0; i < line_in_ellipse.size(); i++)
	{
		for (int j = 0; j < line_in_ellipse[i].line_data.size(); j++)
		{
			cv::Point st, ed;
			st.y = line_in_ellipse[i].line_data[j][0] * smallscale;
			st.x = line_in_ellipse[i].line_data[j][1] * smallscale;
			ed.y = line_in_ellipse[i].line_data[j][2] * smallscale;
			ed.x = line_in_ellipse[i].line_data[j][3] * smallscale;
			cv::line(ImgC, st, ed, cv::Scalar(255, 0, 255), 2);
		}
	}
}


void C2_FTD::drawC2_FTD(cv::Mat &ImgC, int smallscale)
{
	if (ImgC.channels() == 1)
		cv::cvtColor(ImgC, ImgC, cv::COLOR_GRAY2BGR);

	if (isFind)
	{
		cv::RotatedRect temp;
		temp.center.x = landEllipse.center.y * smallscale;
		temp.center.y = landEllipse.center.x * smallscale;
		temp.size.height = landEllipse.size.width * smallscale;
		temp.size.width = landEllipse.size.height * smallscale;
		temp.angle = -landEllipse.angle;
		ellipse(ImgC, temp, cv::Scalar(255, 0, 255), 2);

		cv::Point2f st, ed;
		float l = landEllipse.size.height / 4.0f * 3.0f;
		st.y = -l*landCross[0] + temp.center.y; st.x = -l*landCross[1] + temp.center.x;
		ed.y = l*landCross[0] + temp.center.y; ed.x = l*landCross[1] + temp.center.x;

		st.x *= smallscale, st.y *= smallscale;
		ed.x *= smallscale, ed.y *= smallscale;
		line(ImgC, cv::Point((int)st.x, (int)st.y), cv::Point((int)ed.x, (int)ed.y), cv::Scalar(0, 0, 255), 2);

		st.y = -l*landCross[2] + temp.center.y; st.x = -l*landCross[3] + temp.center.x;
		ed.y = l*landCross[2] + temp.center.y; ed.x = l*landCross[3] + temp.center.x;
		
		st.x *= smallscale, st.y *= smallscale;
		ed.x *= smallscale, ed.y *= smallscale;
		line(ImgC, cv::Point((int)st.x, (int)st.y), cv::Point((int)ed.x, (int)ed.y), cv::Scalar(0, 0, 255), 2);
	}

}


void drawDetectCross(cv::Mat &imgC, cv::RotatedRect &landEllipse, cv::Vec4f &landCross, cv::Scalar elpcolor, cv::Scalar crosscolor, int thickness)
{
	if (imgC.channels() == 1)
		cvtColor(imgC, imgC, cv::COLOR_GRAY2BGR);

	cv::ellipse(imgC, landEllipse, elpcolor, thickness);

	cv::Point2f st, ed;
	float l = (landEllipse.size.height + landEllipse.size.width) / 8.0f * 3.0f;
	st.y = -l*landCross[0] + landEllipse.center.y; st.x = -l*landCross[1] + landEllipse.center.x;
	ed.y = l*landCross[0] + landEllipse.center.y; ed.x = l*landCross[1] + landEllipse.center.x;

	line(imgC, cv::Point((int)st.x, (int)st.y), cv::Point((int)ed.x, (int)ed.y), crosscolor, thickness);

	st.y = -l*landCross[2] + landEllipse.center.y; st.x = -l*landCross[3] + landEllipse.center.x;
	ed.y = l*landCross[2] + landEllipse.center.y; ed.x = l*landCross[3] + landEllipse.center.x;
	line(imgC, cv::Point((int)st.x, (int)st.y), cv::Point((int)ed.x, (int)ed.y), crosscolor, thickness);
}

void drawDetectCross(unsigned char *_imgC, int irows, int icols, double *_landEllipse, double *_landCross, unsigned int *_elpcolor, unsigned int *_crosscolor, int thickness)
{
	cv::Mat imgC(irows, icols, CV_8UC3, _imgC);

	cv::RotatedRect landEllipse;
	landEllipse.center.x = _landEllipse[0];
	landEllipse.center.y = _landEllipse[1];
	landEllipse.size.width = _landEllipse[2];
	landEllipse.size.height = _landEllipse[3];
	landEllipse.angle = _landEllipse[4];

	cv::Vec4f landCross;
	landCross[0] = _landCross[0];
	landCross[1] = _landCross[1];
	landCross[2] = _landCross[2];
	landCross[3] = _landCross[3];

	cv::Scalar elpcolor(_elpcolor[0], _elpcolor[1], _elpcolor[2]);
	cv::Scalar crosscolor(_crosscolor[0], _crosscolor[1], _crosscolor[2]);

	drawDetectCross(imgC, landEllipse, landCross, elpcolor, crosscolor, thickness);
}