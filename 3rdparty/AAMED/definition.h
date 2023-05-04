#pragma once


#define FLED_EDGECONTOURS               0x21
#define FLED_FSAARCCONTOURS             0x22
#define FLED_FSALINES                   0x23
#define FLED_FSAARCSLINKMATRIX          0x24
#define FLED_DPCONTOURS                 0x25

//�����4�����������F��������Զ��C���������
#define FLED_GROUPING_IBmA1_IAnB1      0x00//��ʼ����
#define FLED_GROUPING_FBmA1_FAnB1      0x00//������Զ��β��Զ
#define FLED_GROUPING_FBmA1_CAnB1      0x01//���������β��Զ
#define FLED_GROUPING_CBmA1_FAnB1      0x10//������Զ��β���
#define FLED_GROUPING_CBmA1_CAnB1      0x11//���������β���
#define FLED_GROUPING_CAnB1            0x01//�������
#define FLED_GROUPING_CBmA1            0x10//β���

// ����������У�ȥ���˸�����ɫ�ж��Ƿ����ӵ���������Ϊ����ȥ���ˣ��ں����ȫ�ַ���Ҳ�����ºϲ�һ�ν����жϣ���������kd���㷨����
// ����������⡣


#define FLED_SEARCH_LINKING            true
#define FLED_SEARCH_LINKED             false



// ---------------------�����л�����-----------------------
// ����ӦRDP�㷨
#define ADAPT_APPROX_CONTOURS 1
// ����Ӧ���ʹ����㷨
#define DEFINITE_ERROR_BOUNDED 1
// ��Բ��֤����
#define FASTER_ELLIPSE_VALIDATION 0

// ѡ����Բ�����㷨
#define NONE_CLUSTER_METHOD   0
#define PRASAD_CLUSTER_METHOD 1
#define OUR_CLUSTER_METHOD    2
#define SELECT_CLUSTER_METHOD PRASAD_CLUSTER_METHOD
// ------------------------------------------------------


// �Ƿ���Ҫͳ��ʱ��
// #define DETAIL_BREAKDOWN
