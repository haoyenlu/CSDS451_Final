#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sycl/sycl.hpp>

#include "dpc_common.hpp"

#define COMP(a,b) (abs(a-b) < 0.005)

using namespace std;

class ConvNet{
    private:
        unsigned input_channels;
        unsigned output_channels;
        unsigned stride;
        unsigned kernel;
        unsigned padding;

        int output_chn_block_size;
        int input_chn_block_size;
        int output_width_block_size;

        sycl::queue Q;

        vector<float> weights;
        vector<float> biases;
        float* weights_usm; // weight in share memory
        float* biases_usm;  // bias in share memory 

    public:
        vector<double> naive_time;
        vector<double> reorder_time;
        vector<double> direct_time;
        
    public:
        ConvNet(){};
        ~ConvNet();
        ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel, unsigned padding, unsigned stride);
        vector<vector<float>> forward(vector<vector<float>>& input,int batch_size, int input_height, int input_width);

        void set_weights(vector<float> weights);
        void set_biases(vector<float> biases);
        void set_block_size(int output_chn_block_size,int input_chn_block_size, int output_width_block_size);

        vector<float> add_padding(vector<float> &input,int input_height,int input_width);

        bool cmp_mat(vector<float> &o1, vector<float> &o2,int output_height, int output_width);

        // algorithm
        vector<float> reorder(vector<float> &input,int input_height,int input_width,int output_height, int output_width);
        vector<float> naive(vector<float> &input,int input_height,int input_width,int output_height,int output_width);
        vector<float> direct(vector<float> &input, int input_height,int input_width, int output_height, int output_width);
    

        void device(sycl::queue Q);
};


ConvNet::ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel,unsigned padding, unsigned stride){
    this->input_channels = input_channel;
    this->output_channels = output_channel;
    this->kernel = kernel;
    this->stride = stride;
    this->padding = padding;

    this->weights = vector<float>(output_channel * input_channel * kernel * kernel,0);
    this->biases = vector<float>(output_channel,0);
}

ConvNet::~ConvNet(){
    free(this->weights_usm,this->Q);
    free(this->biases_usm,this->Q);
}

void ConvNet::set_weights(vector<float> weights){
    for (int i=0;i<this->output_channels * this->input_channels * this->kernel * this->kernel;i++){
        this->weights[i] = weights[i];
    }
}

void ConvNet::set_biases(vector<float> biases){
    for (int i=0;i<this->output_channels;i++){
        this->biases[i] = biases[i];      
    }
}

void ConvNet::set_block_size(int output_chn_block_size,int input_chn_block_size, int output_width_block_size){
    this->output_chn_block_size = output_chn_block_size;
    this->input_chn_block_size = input_chn_block_size;
    this->output_width_block_size = output_width_block_size;
}

vector<float> ConvNet::add_padding(vector<float> &input,int input_height,int input_width){
    int input_height_with_padding = input_height + this->padding * 2;
    int input_width_with_padding = input_width + this->padding * 2;

    vector<float> input_with_padding (this->input_channels * input_height_with_padding * input_width_with_padding,0);

    for (int i=0;i<input_channels;i++){
        for (int j=0;j<input_height;j++){
            for (int k=0;k<input_width;k++){
                input_with_padding[(i * input_height + j + this->padding ) * input_width + k +this->padding] = input[( i * input_height + j ) * input_width +  k];
            }
        }
    }

    return input_with_padding;

}

vector<vector<float>> ConvNet::forward(vector<vector<float>> &input, int batch_size , int input_height, int input_width){
    cout << "Forwarding\n";
    int output_height = ((input_height - this->kernel + 2 * this->padding) / this->stride) + 1;
    int output_width = ((input_width - this->kernel + 2 * this->padding) / this->stride) + 1;
    vector<vector<float>> output_batch ( batch_size, vector<float>(this->output_channels * output_height * output_width,0.0));

    int input_height_with_padding = input_height + 2 * this->padding;
    int input_width_with_padding = input_width + 2 * this->padding;

    for (int index = 0; index < batch_size; index ++){
        cout << "----------------(Iteration:" << index << ")----------------\n";
        // add padding 
        vector<float> image_with_padding = this->add_padding(input[index],input_height,input_width);
        // run algorithm
        vector<float> naive_output = this->naive(image_with_padding,input_height_with_padding,input_width_with_padding,output_height,output_width);
        vector<float> reorder_output = this->reorder(image_with_padding,input_height_with_padding,input_width_with_padding,output_height,output_width);
        vector<float> direct_output = this->direct(image_with_padding,input_height_with_padding,input_width_with_padding,output_height,output_width);
        
        // compare result
        this->cmp_mat(naive_output,reorder_output,output_height,output_width);
        this->cmp_mat(naive_output,direct_output,output_height,output_width);
    
        output_batch[index] = direct_output;
    }
    return output_batch;
}

// naive convolution algorithm
vector<float> ConvNet::naive(vector<float> &input,int input_height,int input_width,int output_height,int output_width){
    vector<float> output (this->output_channels * output_height * output_width,0);

    dpc_common::TimeInterval time; // Timing the Convolution Algorithm

    for (int i=0;i<this->input_channels;i++){
        for (int j=0;j<this->output_channels;j++){
            for (int k=0;k<output_width;k++){
                for (int l=0;l<output_height;l++){
                    for (int m=0;m<this->kernel;m++){
                        for(int n=0;n<this->kernel;n++){
                            int input_row = l * this->stride + n;
                            int input_col = k * this->stride + m;
                            if (input_row >= input_height || input_col >= input_width) break;
                            output[(j * output_height + l) * output_width + k] += input[(i * input_height + input_row) * input_width + input_col] * this->weights[((j * this->input_channels + i ) * this->kernel + n) * this->kernel + m];            
                        }
                    }
                }
            }
        }
    }

    
    double time_elapsed = time.Elapsed();
    cout << "Naive Convolution Time: " << time_elapsed << endl;

    this->naive_time.push_back(time_elapsed);

    return output;
}


// reorder the loop of naive convolution algorithm
vector<float> ConvNet::reorder(vector<float> &input, int input_height, int input_width, int output_height, int output_width){
    
    vector<float> output (this->output_channels * output_height * output_width,0);

    dpc_common::TimeInterval time; // Timing the Convolution Algorithm

    for (int l=0;l<output_height;l++){ // H_o ( output height )
        for (int n=0;n<this->kernel;n++){ // H_f ( kernel height )
            for (int m=0;m<this->kernel;m++){ // W_f ( kernel width )
                for (int i=0;i<this->input_channels;i++){ //C_i ( input channel )
                    for (int k=0;k<output_width;k++){ // W_o ( output width )
                        for (int j=0;j<this->output_channels;j++){ // C_o ( output channel )
                            int input_row = l * this->stride + n;
                            int input_col = k * this->stride + m;
                            if (input_row >= input_height || input_col >= input_width) break;
                            output[(j * output_height + l) * output_width + k] += input[(i * input_height + input_row) * input_width + input_col] * this->weights[((j * this->input_channels + i ) * this->kernel + n) * this->kernel + m];             
                        }
                    }
                }
            }
        }
    }
    
    double time_elapsed = time.Elapsed();
    cout << "Reorder Convolution Time: " << time_elapsed << endl;

    this->reorder_time.push_back(time_elapsed);

    return output;
}


vector<float> ConvNet::direct(vector<float> &input, int input_height,int input_width, int output_height,int output_width){

    
    vector<float> output (this->output_channels * output_height * output_width,0);
    sycl::buffer<float,1> output_buf(output.data(),output.size());
    sycl::buffer<float,1> input_buf(input.data(),input.size());
    
    int output_channels = this->output_channels;
    int input_channels = this->input_channels;
    int stride = this->stride;
    int kernel = this->kernel;

    
    float *weights_share = this->weights_usm;
    float *biases_share = this->biases_usm;

    int output_chn_block_size = this->output_chn_block_size;
    int input_chn_block_size = this->input_chn_block_size;
    int output_width_block_size = this->output_width_block_size;

    int output_chn_block_cnt = static_cast<int>(output_channels / output_chn_block_size );
    int input_chn_block_cnt = static_cast<int>(input_channels / input_chn_block_size);
    int output_width_block_cnt = static_cast<int>(output_width / output_width_block_size);

    //cout << output_chn_block_cnt << "," << input_chn_block_cnt << "," << output_width_block_cnt << endl;

    sycl::range<2> two_layer_paral_range(output_chn_block_cnt,output_width_block_cnt); // parallel output channel and output width
    sycl::range<1> one_layer_paral_range(output_chn_block_cnt); // parallel output channel only

    dpc_common::TimeInterval time; // Timing the Convolution Algorithm

    this->Q.submit([&](sycl::handler &h){
        auto output_share = output_buf.get_access<sycl::access::mode::read_write>(h);
        auto input_share = input_buf.get_access<sycl::access::mode::read_write>(h);


        h.parallel_for(two_layer_paral_range,[=](sycl::id<2> index){
            int j = index[0];
            int k = index[1];
            for (int i=0;i<input_chn_block_cnt;i++){
                for (int l=0;l<output_height;l++){
                    //for (int k=0;k<output_width_block_cnt;k++){
                        for (int n=0;n<kernel;n++){
                            int block_input_height = l * stride + n;
                            if (block_input_height >= input_height) break;

                    
                            for (int m=0;m<kernel;m++){
                                for (int ii=0;ii<input_chn_block_size;ii++){
                                    int block_input_chn = i * input_chn_block_size + ii;
                                    if (block_input_chn >= input_channels) break;

                                    for (int kk=0;kk<output_width_block_size;kk++){
                                        int block_output_width = k * output_width_block_size  + kk;
                                        int block_input_width = stride * block_output_width  + m;
                                        if (block_output_width >= output_width || block_input_width >= input_width) break;

                                        for (int jj=0;jj<output_chn_block_size;jj++){
                                            int block_output_chn = j * output_chn_block_size  + jj; 
                                            if (block_output_chn >= output_channels) break;
                                            float ans = input_share[(block_input_chn* input_height + block_input_height) * input_width + block_input_width] * weights_share[((block_output_chn * input_channels + block_input_chn) * kernel + n ) * kernel + m];

                                            output_share[(block_output_chn * output_height + l) * output_width + block_output_width] += ans;
                                        }       
                                    }
                                }
                            }
                        }
                    //}
                }
            }
        });
    });
    this->Q.wait();

    double time_elapsed = time.Elapsed();
    cout << "Direct Convolution Time: " << time_elapsed << endl;

    this->direct_time.push_back(time_elapsed);

    /*
    sycl::host_accessor<float,1> output_host(output_buf);
    for (int i=0;i<this->output_channels * output_height * output_width;i++){
        output[i] = output_host[i];
    }
    */

    return output;
}


bool ConvNet::cmp_mat(vector<float> &o1, vector<float> &o2,int output_height , int output_width){
    // compare value
    bool same = true;
    int number_of_difference = 0;
    for (int i=0;i<this->output_channels * output_height * output_width;i++){
        if (!COMP(o1[i],o2[i])){
            if (same) {
                printf("Two matrix have different values at %d: ( %f, %f)\n",i,o1[i],o2[i]);
                same = false;
            }
            number_of_difference ++;
        }
    }
    cout << "Number of difference: "<< number_of_difference << endl;
    return same;
}

void ConvNet::device(sycl::queue Q){
    this->Q = Q;
    cout << "Device: " << this->Q.get_device().get_info<sycl::info::device::name>() << "\n";

    this->weights_usm = sycl::malloc_shared<float>(this->output_channels * this->input_channels * this->kernel * this->kernel,this->Q);
    for (int i=0;i<this->output_channels * this->input_channels * this->kernel * this->kernel ;i++){
        this->weights_usm[i] = this->weights[i];
    }

    this->biases_usm = sycl::malloc_shared<float>(this->output_channels,this->Q);
    for (int i=0;i<this->output_channels;i++){
        this->biases_usm[i] = this->biases[i];
    }
}

vector<vector<float>> numpy_text_to_batch(int batch_size, int input_channels, int height, int width, string filepath){
    cout << "Reading file: " << filepath << endl;
    vector<vector<float>> batch = vector<vector<float>> ( batch_size, vector<float>(input_channels * height* width,0));

    ifstream file;
    file.open(filepath,ios::in);

    if (!file.is_open()){
        cout<< "Error: Cannot open file " << filepath <<endl;
        return batch;
    }
    
    for (int i=0;i<batch_size;i++){
        for (int j=0;j<input_channels * height * width; j++){
            float temp = 1;
            file >> temp;
            batch[i][j] = temp;

            if ( ! file ) {
                cout << "Error reading file for element " << i << "," << j << endl; 
            }
        }

    }

    return batch;
}

vector<float> numpy_text_to_weight(int output_channels, int input_channels, int kernel_size, string filepath){
    cout << "Reading file: " << filepath << endl;

    vector<float> weight (output_channels * input_channels * kernel_size * kernel_size,0);

    ifstream file;
    file.open(filepath,ios::in);

    if (!file.is_open()){
        cout<< "Error: Cannot open file " <<  filepath << endl;
        return weight;
    }
    
    for (int i=0;i<output_channels * input_channels * kernel_size * kernel_size;i++){
        file >> weight[i];
        if ( ! file ) {
            cout << "Error reading file for element " << i << endl; 
        }
    }


    return weight;
}


vector<float> numpy_text_to_bias(int output_channels,string filepath){
    cout << "Reading file: " << filepath << endl;
    
    vector<float> bias (output_channels,0);

    ifstream file;
    file.open(filepath,ios::in);

    if (!file.is_open()){
        cout << "Error: Cannot open file "<< filepath << endl;
        return bias;
    }

    for (int i=0;i<output_channels;i++){
        file >> bias[i];
        if (!file){
            cout << "Error reading file for element " << i << endl; 
        }
    }   

    return bias;
}

void write_performance_time_to_csv(vector<double> naive_time,vector<double> reorder_time, vector<double> direct_time, string output_filename){
    cout << "Write performance to " << output_filename << endl; 
    ofstream file;
    file.open(output_filename);
    file << "naive,reorder,direct\n"; // first row in the csv

    int batch_size = naive_time.size();
    for (int i=0;i<batch_size;i++){
        file << naive_time[i] << ',' << reorder_time[i] << ',' << direct_time[i] << "\n";
    }
}


int main(){

    string layers[14] = {"layer0","layer1","layer2","layer3","layer4","layer5","layer6","layer7","layer8","layer9","layer10","layer11","layer12","layer13"};

    int input_channels[13] = {3,64,64,128,128,256,256,256,512,512,512,512,512};
    int output_channels[13] = {64,64,128,128,256,256,256,512,512,512,512,512,512};
    int input_heights[13] = {128,128,64,64,32,32,32,16,16,16,8,8,8};
    int input_widths[13] = {128,128,64,64,32,32,32,16,16,16,8,8,8};

    int output_chn_block_size = 1;
    int input_chn_block_size[13] = {1,16,16,32,32,32,32,32,32,32,32,32,32};
    int output_width_block_size[13] = {32,32,16,16,8,8,8,8,8,8,4,4,4};

    int batch_size = 32;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    sycl::queue q(sycl::default_selector_v);

    for (int i=0;i<13;i++){
        cout << "---------------------" << "Running " << layers[i+1] << "---------------------" << endl;
        //batch 
        const string batch_filepath = "batches/" +  layers[i] + "_" + to_string(batch_size) + 'x' + to_string(input_channels[i]) + 'x' + to_string(input_heights[i]) + 'x' + to_string(input_widths[i]) + ".txt";
        // weight
        const string weight_filepath = "weights/" + layers[i+1]+ "_" + to_string(output_channels[i]) + 'x' + to_string(input_channels[i]) + 'x' + to_string(kernel_size) + 'x' + to_string(kernel_size) + ".txt";
        // bias
        const string bias_filepath = "biases/" + layers[i+1] + "_" + to_string(output_channels[i]) + ".txt";


        vector<vector<float>> input = numpy_text_to_batch(batch_size,input_channels[i],input_heights[i],input_widths[i],batch_filepath);
        vector<float> weights = numpy_text_to_weight(output_channels[i],input_channels[i],kernel_size,weight_filepath);
        vector<float> biases = numpy_text_to_bias(output_channels[i],bias_filepath);



        ConvNet model(input_channels[i],output_channels[i],kernel_size,padding,stride);
        model.set_weights(weights);
        model.set_biases(biases);
        model.set_block_size(output_chn_block_size,input_chn_block_size[i],output_width_block_size[i]);

        model.device(q);
        vector<vector<float>> output = model.forward(input,batch_size,input_heights[i],input_widths[i]);
            
        for (int i=0 ;i<10;i++) cout << output[0][i] << " ";
        cout << endl;

        string output_filename = "csv_new3/" + layers[i+1]+ "_" + to_string(input_channels[i]) + 'x' + to_string(output_channels[i]) + ".csv";
        write_performance_time_to_csv(model.naive_time,model.reorder_time,model.direct_time,output_filename);
    }

 
    return 0;
}