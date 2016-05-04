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
	Caffe::set_mode(Caffe::GPU);
	
	string alt_param_file = "examples/stochastic_depth_caffe/alt_solver.prototxt";
	SolverParameter alt_param;
 	ReadSolverParamsFromTextFileOrDie(alt_param_file, &alt_param);
    Solver<float>* alt_solver = SolverRegistry<float>::CreateSolver(alt_param); 
    shared_ptr<Net<float> > alt_net = alt_solver->net();
    vector<shared_ptr<Layer<float> > > alt_layers = alt_net->layers();
	
	cout <<"alt_layers size: " << alt_layers.size() << endl;
	for (int i = 0; i < alt_layers.size(); i++) {
//		cout << alt_layers[i]->type() << endl;
	}
}
