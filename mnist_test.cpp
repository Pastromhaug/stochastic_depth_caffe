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
	
	string param_file = "examples/mnist/lenet_solver.prototxt";
	
  	SolverParameter param;
  	ReadSolverParamsFromTextFileOrDie(param_file, &param);
  	Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);	
	shared_ptr<Net<float> > net = solver->net();
	vector<shared_ptr<Layer<float> > > layers = net->layers();
	
	cout << "layers size: " << layers.size() << endl;
	for (int i = 0; i < layers.size(); i++) {
		cout << layers[i]->type()  << endl; 
		//net->ForwardFromTo(i, i+1)	
	}
	
	vector<shared_ptr<Blob<float> > > blobs = net->blobs();
	
	cout << "blobs size: " <<  blobs.size() << endl;	
	for (int i = 0; i < blobs.size(); i++) {
		//cout << blobs[i] << endl;
	}

	vector<vector<Blob<float>* > > bottom_vecs = net->bottom_vecs();
	cout << "bottom vecs size: " <<  bottom_vecs.size() << endl;
	for (int i = 0; i < bottom_vecs.size(); i++) {
		//cout << bottom_vecs[i].size() << endl;	
	}
	
//	cout << "iter: ";
//	solver->Solve();
//	int iter = solver->iter();
//	cout << iter << endl;
//	
//	float loss = net->ForwardFromTo(0, layers.size()-1);
//	cout << "loss: ";
//	cout << loss << endl;
//
//	net->BackwardFromTo(layers.size()-1, 0);
//	net->Update();


	

//	loss = net->ForwardFromTo(0, layers.size()-1);
//	cout << "loss: ";
//	cout << loss << endl;
//	cout << "yeee\n";	
	
	
}

//template <typename Dtype>
//void Solver<Dtype>::Step(int iters) {
//  const int start_iter = iter_;
//  const int stop_iter = iter_ + iters;
//  int average_loss = this->param_.average_loss();
//  losses_.clear();
//  smoothed_loss_ = 0;
//
//    // zero-init the params
//    net_->ClearParamDiffs();
//    if (param_.test_interval() && iter_ % param_.test_interval() == 0
//        && (iter_ > 0 || param_.test_initialization())
//        && Caffe::root_solver()) {
//      TestAll();
//    }
//
//    for (int i = 0; i < callbacks_.size(); ++i) {
//      callbacks_[i]->on_start();
//    }
//    const bool display = param_.display() && iter_ % param_.display() == 0;
//    net_->set_debug_info(display && param_.debug_info());
//    // accumulate the loss and gradient
//    Dtype loss = 0;
//    loss += net_->ForwardBackward();
//	cout << "loss: " << loss << endl;
//}
