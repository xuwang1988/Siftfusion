#include <windows.h>

#include"read_input.h"
using namespace cv;

Mat readValuesFromTxt(char* path){
	Mat K = Mat::zeros(3,3,CV_64F);
	FILE *fp;
	fopen_s(&fp,path,"r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open input file in.list!\n");
		exit(1);
	}
	float in[9];
	int i = 0;
	while (!feof(fp)) {
		fscanf_s(fp, "%f", &in[i++]);		
	}
	for (int j = 0; j < 3; ++j)
		for (int k = 0; k < 3; ++k)
			K.at<double>(j, k) = in[3*j+k];

	return K;
}

std::vector<string> dirSmart(char* path){
	vector<string> files;
	WIN32_FIND_DATA search_data;
	memset(&search_data, 0, sizeof(WIN32_FIND_DATA));
	DWORD dwNum = MultiByteToWideChar(CP_ACP, 0, path, -1, NULL, 0);
	wchar_t *dir;
	dir = new wchar_t[dwNum];
	MultiByteToWideChar(CP_ACP, 0, path, -1, dir, dwNum);
	HANDLE handle = FindFirstFile(dir, &search_data);
	while (handle != INVALID_HANDLE_VALUE)
	{
		char ch[260];
		char DefChar = ' ';
		WideCharToMultiByte(CP_ACP, 0, search_data.cFileName, -1, ch, 260, &DefChar, NULL);
		//files[i] = new char[260];
		files.push_back(ch);
		if (FindNextFile(handle, &search_data) == FALSE)
			break;
	}

	return files;
}