#include <sycl/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/random>
#include <fstream>
#include <string>
#include <random>
#include <vector>

#include "mkl.h"
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

class ConvolutionLayer{
    public :
    ConbolutionLayer(int input_channels,int output_channels,int kernel_size, int stride,int padding){
        this->input_channels = input_channels;
        this->output_channels = output_channels;
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;

        this->weights = vector<vector<vector<float>>> (output_channels,vector<vector<float>> (input_channels,vector<float> (kernel_size * kernel_size)));
        this->biases = vector<float> (output_channels);
    }

    void set_weight(vector<vector<vector<float>>> weights){
        for (int i=0;i<this.output_channels,i++){
            for (int j=0;j<this.input_channels;j++){
                for (int k=0;k<this.kernel_size * kernel_size;k++){
                    this->weights[i][j][k] = weights[i][j][k];
                }
            }
        }
    }

    void set_bias(vector<float> biases){
        for (int i=0;i<this.output_channels;i++){
            this->biases = biases[i];
        }
    }
    

    // naive method
    vector<vector<float>> forward(vector<vector<vector<float>>> input){
        int input_channels = input.size();
        int image_height = input[0].size();
        int image_width = input[0][0].size();

        //check if input channel is the same as specified
        cout << "Input shape:(" << input_channels << "," << image_height << "," << image_width <<")" << endl; 

    }

}
