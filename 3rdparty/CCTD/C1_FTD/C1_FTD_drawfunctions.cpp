#include "C1_FTD/C1_FTD.h"

void C1_FTD::drawCrossFeatures(cv::Mat &imgG, int smallscale)
{
	if(imgG.channels() == 1)
		cv::cvtColor(imgG, imgG, cv::COLOR_GRAY2BGR);
	
	int l = 10;
	for (int i = 0; i < cross_feature.size(); i++)
	{
		if (cross_feature[i].isValid == false)
			continue;
		circle(imgG, cv::Point((int)cross_feature[i].O.y * smallscale, (int)cross_feature[i].O.x * smallscale), 2, cv::Scalar(0, 0, 255), 2);
		cv::line(imgG, cv::Point((int)cross_feature[i].O.y * smallscale, (int)cross_feature[i].O.x * smallscale), 
					   cv::Point((int)(cross_feature[i].O.y * smallscale + l*cross_feature[i].V_l.y * smallscale), (int)(cross_feature[i].O.x * smallscale + l*cross_feature[i].V_l.x * smallscale)), cv::Scalar(255, 0, 0), 2);
		cv::line(imgG, cv::Point((int)cross_feature[i].O.y * smallscale, (int)cross_feature[i].O.x * smallscale), cv::Point((int)(cross_feature[i].O.y * smallscale + l*cross_feature[i].V_r.y * smallscale), (int)(cross_feature[i].O.x * smallscale + l*cross_feature[i].V_r.x * smallscale)), cv::Scalar(255, 0, 255), 2);
	}

}

void C1_FTD::drawCrossFeatures(cv::Mat &ImgG, bool isShow)
{
	if (isShow)
	{
		cv::Mat ImgT = ImgG.clone();
		
		cv::cvtColor(ImgT, ImgT, cv::COLOR_GRAY2BGR);
		int l = 10;
		for (int i = 0; i < cross_feature.size(); i++)
		{
			if (cross_feature[i].isValid == false)
				continue;
			circle(ImgT, cv::Point((int)cross_feature[i].O.y, (int)cross_feature[i].O.x), 2, cv::Scalar(0, 0, 255), 2);

			cv::line(ImgT, cv::Point((int)cross_feature[i].O.y, (int)cross_feature[i].O.x), cv::Point((int)(cross_feature[i].O.y + l*cross_feature[i].V_l.y), (int)(cross_feature[i].O.x + l*cross_feature[i].V_l.x)), cv::Scalar(255, 0, 0), 2);

			cv::line(ImgT, cv::Point((int)cross_feature[i].O.y, (int)cross_feature[i].O.x), cv::Point((int)(cross_feature[i].O.y + l*cross_feature[i].V_r.y), (int)(cross_feature[i].O.x + l*cross_feature[i].V_r.x)), cv::Scalar(255, 0, 255), 2);

		}
		cv::imshow("Cross Features", ImgT);
		//cv::imwrite("CrossFeatures.png", ImgT);
		cv::waitKey(1);
	}
	else
	{
		if(ImgG.channels() == 1)
			cv::cvtColor(ImgG, ImgG, cv::COLOR_GRAY2BGR);
		int l = 10;
		for (int i = 0; i < cross_feature.size(); i++)
		{
			if (cross_feature[i].isValid == false)
				continue;
			circle(ImgG, cv::Point((int)cross_feature[i].O.y, (int)cross_feature[i].O.x), 2, cv::Scalar(0, 0, 255), 2);

			cv::line(ImgG, cv::Point((int)cross_feature[i].O.y, (int)cross_feature[i].O.x), cv::Point((int)(cross_feature[i].O.y + l*cross_feature[i].V_l.y), (int)(cross_feature[i].O.x + l*cross_feature[i].V_l.x)), cv::Scalar(255, 0, 0), 2);

			cv::line(ImgG, cv::Point((int)cross_feature[i].O.y, (int)cross_feature[i].O.x), cv::Point((int)(cross_feature[i].O.y + l*cross_feature[i].V_r.y), (int)(cross_feature[i].O.x + l*cross_feature[i].V_r.x)), cv::Scalar(255, 0, 255), 2);

		}
	}
	
}

void C1_FTD::drawC1_FTD(cv::Mat &ImgG, bool isShow)
{
	if (isShow)
	{
		cv::Mat ImgT = ImgG.clone();
		cv::cvtColor(ImgT, ImgT, cv::COLOR_GRAY2BGR);
		cv::Point st, ed;
		cv::circle(ImgT, cv::Point((int)crossCenter.y, (int)crossCenter.x), 2, cv::Scalar(0, 255, 255), 3);

		cv::line(ImgT, cv::Point((int)cross4Points[0].y, (int)cross4Points[0].x), cv::Point((int)cross4Points[1].y, (int)cross4Points[1].x), cv::Scalar(0, 0, 255), 3);
		cv::line(ImgT, cv::Point((int)cross4Points[1].y, (int)cross4Points[1].x), cv::Point((int)cross4Points[2].y, (int)cross4Points[2].x), cv::Scalar(0, 0, 255), 3);
		cv::line(ImgT, cv::Point((int)cross4Points[2].y, (int)cross4Points[2].x), cv::Point((int)cross4Points[3].y, (int)cross4Points[3].x), cv::Scalar(0, 0, 255), 3);
		cv::line(ImgT, cv::Point((int)cross4Points[3].y, (int)cross4Points[3].x), cv::Point((int)cross4Points[0].y, (int)cross4Points[0].x), cv::Scalar(0, 0, 255), 3);

		cv::imshow("C1_FTD", ImgT);
		cv::waitKey(1);
	}
	else
	{
		if(ImgG.channels() == 1)
			cv::cvtColor(ImgG, ImgG, cv::COLOR_GRAY2BGR);
		cv::Point st, ed;
		cv::circle(ImgG, cv::Point((int)crossCenter.y, (int)crossCenter.x), 2, cv::Scalar(0, 255, 255), 3);

		cv::line(ImgG, cv::Point((int)cross4Points[0].y, (int)cross4Points[0].x), cv::Point((int)cross4Points[1].y, (int)cross4Points[1].x), cv::Scalar(0, 0, 255), 3);
		cv::line(ImgG, cv::Point((int)cross4Points[1].y, (int)cross4Points[1].x), cv::Point((int)cross4Points[2].y, (int)cross4Points[2].x), cv::Scalar(0, 0, 255), 3);
		cv::line(ImgG, cv::Point((int)cross4Points[2].y, (int)cross4Points[2].x), cv::Point((int)cross4Points[3].y, (int)cross4Points[3].x), cv::Scalar(0, 0, 255), 3);
		cv::line(ImgG, cv::Point((int)cross4Points[3].y, (int)cross4Points[3].x), cv::Point((int)cross4Points[0].y, (int)cross4Points[0].x), cv::Scalar(0, 0, 255), 3);
	}
	
}

void C1_FTD::drawC1_FTD(cv::Mat &imgG, int smallscale)
{
	if(imgG.channels() == 1)
		cv::cvtColor(imgG, imgG, cv::COLOR_GRAY2BGR);

	cv::Point st, ed;
	cv::circle(imgG, cv::Point((int)crossCenter.y * smallscale, (int)crossCenter.x * smallscale), 2, cv::Scalar(0, 255, 255), 3);

	cv::line(imgG, cv::Point((int)cross4Points[0].y * smallscale, (int)cross4Points[0].x * smallscale), 
				   cv::Point((int)cross4Points[1].y * smallscale, (int)cross4Points[1].x * smallscale), cv::Scalar(0, 0, 255), 3);
	cv::line(imgG, cv::Point((int)cross4Points[1].y * smallscale, (int)cross4Points[1].x * smallscale), 
	               cv::Point((int)cross4Points[2].y * smallscale, (int)cross4Points[2].x * smallscale), cv::Scalar(0, 0, 255), 3);
	cv::line(imgG, cv::Point((int)cross4Points[2].y * smallscale, (int)cross4Points[2].x * smallscale), 
				   cv::Point((int)cross4Points[3].y * smallscale, (int)cross4Points[3].x * smallscale), cv::Scalar(0, 0, 255), 3);
	cv::line(imgG, cv::Point((int)cross4Points[3].y * smallscale, (int)cross4Points[3].x * smallscale), 
				   cv::Point((int)cross4Points[0].y * smallscale, (int)cross4Points[0].x * smallscale), cv::Scalar(0, 0, 255), 3);
}