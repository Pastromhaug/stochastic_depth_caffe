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
	LOG(INFO) << argv[0] << " [GPU] [Device ID]";
	
    //Setting CPU or GPU
	if (argc >= 2 && strcmp(argv[1], "GPU") == 0)
	{
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		if (argc == 3)
        {
      		device_id = atoi(argv[2]);
    	}
		Caffe::SetDevice(device_id);
   		LOG(INFO) << "Using GPU #" << device_id;
	}
	else
	{
    	LOG(INFO) << "Using CPU";
    	Caffe::set_mode(Caffe::CPU);
	}
	
	// Set to TEST Phase
	Caffe::set_phase(Caffe::TEST);
	
	// Load net
	// Assume you are in Caffe master directory
	Net<float> net("./examples/prediction_example/prediction_example.prototxt");
}
