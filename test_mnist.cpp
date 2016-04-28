#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "/home/zl499/caffe/include/caffe/caffe.hpp"
#include "/home/zl499/caffe/include/caffe/util/io.hpp"
#include "/home/zl499/caffe/include/caffe/blob.hpp"

using namespace caffe;
using namespace std;


int main(int argc, char** argv)
{
	std::cout << "it's working\n";
	Caffe::set_mode(Caffe::GPU);
}
