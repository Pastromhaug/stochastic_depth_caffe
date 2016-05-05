#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"


using namespace caffe;
using namespace std;

void standardResLayer(int & elts, int & idx, vector<int>* layers_chosen, double ran, double prob);
void transitionResLayer(int & elts, int & idx, vector<int>* layers_chosen, double ran, double prob);


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


	vector<int>* layers_chosen = new vector<int>();
	solver->ChooseLayers_StochDep(layers_chosen);

	for (int i = 0; i < layers_chosen->size(); i++) {
		cout << (*layers_chosen)[i] << ": " <<layers[(*layers_chosen)[i]]->type() << endl;
	}
//	solver->Solve_StochDep();
}

void standardResLayer(int & elts, int & idx, vector<int>* layers_chosen, double ran, double prob) {  
	cout << prob << endl;
   	if (ran < prob){ // include res block
    	for (int i = 0; i < 10; i++){
			(*layers_chosen)[idx] = elts;
			elts += 1;
			idx += 1;
       	}
 	}
   	else{  // skip res block
		(*layers_chosen)[idx] = elts;
		elts += 10;
		idx += 1;
    }
}

void transitionResLayer(int & elts, int& idx, vector<int>* layers_chosen, double ran, double prob){
	cout << prob << endl;
   	if (ran < prob) { //include res block
       	for (int i = 0; i < 13; i++) { 
			(*layers_chosen)[idx] = elts;
			elts += 1;
			idx += 1;
      	}   
  	}
  	else { // skip res block
		(*layers_chosen)[idx] = elts;
		elts += 2;
		idx += 1;
       	
		(*layers_chosen)[idx] = elts;
		elts += 1;
		idx += 1;
      	
		(*layers_chosen)[idx] = elts;
		elts += 1;
		idx += 1;
    
	   	(*layers_chosen)[idx] = elts;
		elts += 9;
		idx += 1;
 	}   
}



template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo_StochDep(vector<int>* layers_chosen) {
	int start = 0;
	int end = layers_.size() - 1;
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
	int start = layers_.size() - 1; 
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
Dtype Net<Dtype>::ForwardBackward_StochDep(vector<int>* layers_chosen) {
    Dtype loss;
    Forward_StochDep(layers_chosen, &loss);
    Backward_StochDep(layers_chosen);
    return loss;
}

template<typename Dtype>
void Solver<Dtype>::ChooseLayers_StochDep(vector<int>* layers_chosen){
	layers_chosen->resize(this->net()->layers().size());
	int elts = 0;   
    int idx = 0;
	for (int i = 0; i < 4; i++){
		(*layers_chosen)[idx] = elts;
		elts += 1;
		idx += 1;
    }

    srand(time(NULL));
        
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)0)/13);
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)1)/13);
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)2)/13);
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)3)/13);

    transitionResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)4)/13);
        
    standardResLayer(elts, idx,  layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)5)/13);
    standardResLayer(elts, idx,  layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)6)/13);
    standardResLayer(elts, idx,  layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)7)/13);
    standardResLayer(elts, idx,  layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)8)/13);
        
    transitionResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)9)/13);
        
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)10)/13);
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)11)/13);
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)12)/13);
    standardResLayer(elts, idx, layers_chosen, (double) rand()/RAND_MAX, 1 - 0.5*((double)13)/13);

    for (int i = 0; i < 4; i++) {
        (*layers_chosen)[idx] = elts;
		elts += 1;
		idx += 1;
    }
	layers_chosen->resize(idx);
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


template <typename Dtype>
void Solver<Dtype>::Step_StochDep(int iters, vector<int>* layers_chosen) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    
    for (int i = 0; i < param_.iter_size(); ++i) {
      ChooseLayers_StochDep(layers_chosen);
      loss += net_->ForwardBackward_StochDep(layers_chosen);
    }
    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}


template <typename Dtype>
void Solver<Dtype>::Solve_StochDep(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  vector<int>* layers_chosen = new vector<int>();
  Step_StochDep(param_.max_iter() - iter_, layers_chosen);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

