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
	
	string param_file = "examples/stochastic_depth_caffe/solver.prototxt";
  	SolverParameter param;
  	ReadSolverParamsFromTextFileOrDie(param_file, &param);
  	Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);	
	shared_ptr<Net<float> > net = solver->net();
	vector<shared_ptr<Layer<float> > > layers = net->layers();
	
	cout << "layers size: " <<  layers.size() << endl;
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
	cout << "bottom_vecs size: " << bottom_vecs.size() << endl;
	for (int i = 0; i < bottom_vecs.size(); i++) {
		//cout << bottom_vecs[i].size() << endl;	
	}
}



template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo_StochDep(vector<int>* layers_chosen) {
	int start = 0;
	int end = 0;
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo_StochDep(vector<int>* layers_chosen) {
	int start = 0; 
	int end = 0;
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
  }
}

template<typename Dtype>
Dtype Net<Dtype>::ForwardBackward_StochDep() {
	vector<int>* layers_chosen = ChooseLayers_StochDep();
    Dtype loss;
    Forward_StochDep(layers_chosen, &loss);
    Backward_StochDep(layers_chosen);
    return loss;
}

template <typename Dtype>
vector<int>* Net<Dtype>::ChooseLayers_StochDep(){
	vector<int>* layers_chosen(10);
	return layers_chosen;
}

template <typename Dtype>
void Net<Dtype>::Backward_StochDep( vector<int>* layers_chosen) {
  BackwardFromTo_StochDep(layers_chosen);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward_StochDep(vector<int>* layers_chosen, Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo_StochDep(layers_chosen);
  } else {
    ForwardFromTo_StochDep(layers_chosen);
  }
  return net_output_blobs_;
}
