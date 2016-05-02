#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"

using namespace caffe;
using namespace std;


int main(int argc, char** argv)
{
	std::cout << "it's working\n";
	Caffe::set_mode(Caffe::CPU);
	
	string param_file = "examples/stochastic_depth_caffe/solver.prototxt";
	
  	SolverParameter param;
  	ReadSolverParamsFromTextFileOrDie(param_file, &param);
  	Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);	
	shared_ptr<Net<float> > net = solver->net();
	vector<shared_ptr<Layer<float> > > layers = net->layers();
	
	for (int i = 0; i < layers.size(); i++) {
		cout << layers[i]  << endl;
	}
	
	cout << "yeee\n";	

}
