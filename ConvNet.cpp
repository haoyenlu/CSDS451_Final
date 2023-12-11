#include <vector>
#include <iostream>
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

        vector<float> add_padding(vector<float> input,int input_height,int input_width);

        bool cmp_mat(vector<float> o1, vector<float> o2,int output_height, int output_width);

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

vector<float> ConvNet::add_padding(vector<float> input,int input_height,int input_width){
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
    int output_height = ((input_height - this->kernel + 2 * this->padding) / this->stride) + 1;
    int output_width = ((input_width - this->kernel + 2 * this->padding) / this->stride) + 1;
    vector<vector<float>> output_batch ( batch_size, vector<float>(this->output_channels * output_height * output_width,0.0));

    input_height = input_height + 2 * this->padding;
    input_width = input_width + 2 * this->padding;

    for (int index = 0; index < batch_size; index ++){

        // add padding 
        vector<float> image_with_padding = this->add_padding(input[index],input_height,input_width);
        // run algorithm
        vector<float> naive_output = this->naive(image_with_padding,input_height,input_width,output_height,output_width);
        vector<float> reorder_output = this->reorder(image_with_padding,input_height,input_width,output_height,output_width);
        vector<float> direct_output = this->direct(image_with_padding,input_height,input_width,output_height,output_width);
        
        // compare result
        this->cmp_mat(naive_output,reorder_output,output_height,output_width);
        this->cmp_mat(reorder_output,direct_output,output_height,output_width);
    
        output_batch[index] = reorder_output;
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

    float *output_share = sycl::malloc_shared<float>(this->output_channels * output_height * output_width,this->Q);
    for (int i=0;i<this->output_channels * output_height * output_width;i++){
        output_share[i] = 0;
    }

    float *input_share = sycl::malloc_shared<float>(this->input_channels * input_height * input_width,this->Q);
    for (int i=0;i<this->input_channels * input_height * input_width ;i++){
        input_share[i] = input[i];
    }
    
    float *weights_share = this->weights_usm;
    float *biases_share = this->biases_usm;

    dpc_common::TimeInterval time; // Timing the Convolution Algorithm

    this->Q.submit([&](sycl::handler &h){
        int output_channels = this->output_channels;
        int input_channels = this->input_channels;
        int stride = this->stride;
        int kernel = this->kernel;

        int output_chn_block_size = 1;
        int input_chn_block_size = 16;
        int output_width_block_size = 16;

        int output_chn_block_cnt = static_cast<int>(output_channels / output_chn_block_size );
        int input_chn_block_cnt = static_cast<int>(input_channels / input_chn_block_size);
        int output_width_block_cnt = static_cast<int>(output_width / output_width_block_size);

        h.parallel_for(sycl::range<1>(output_chn_block_cnt),[=](sycl::id<1> j){
            for (int i=0;i< input_chn_block_cnt;i++){
                for (int l=0;l<output_height;l++){
                    for (int k=0;k<output_width_block_cnt;k++){
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
                    }
                }
            }
        });
    });
    this->Q.wait();

    double time_elapsed = time.Elapsed();
    cout << "Direct Convolution Time: " << time_elapsed << endl;

    this->direct_time.push_back(time_elapsed);

    vector<float> output_result (this->output_channels * output_height * output_width,0);
    for (int i=0;i<this->output_channels * output_height * output_width;i++){
        output_result[i] = output_share[i];
    }

    free(output_share,this->Q);
    free(input_share,this->Q);

    return output_result;
}


bool ConvNet::cmp_mat(vector<float> o1, vector<float> o2,int output_height , int output_width){
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
            file >> batch[i][j];
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
    ofstream file;
    file.open(output_filename);
    file << "naive,reorder,direct\n"; // first row in the csv

    int batch_size = naive_time.size();
    for (int i=0;i<batch_size;i++){
        file << naive_time[i] << ',' << reorder_time[i] << ',' << direct_time[i] << "\n";
    }
}


int main(){

    // layer 2 convolution
    int batch_size = 32;
    int input_channels = 128;
    int output_channels = 128;
    int input_height = 16;
    int input_width = 16;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;


    const string batch_filepath = "batches/layer3_batch_" + to_string(batch_size) + 'x' + to_string(input_channels) + 'x' + to_string(input_height) + 'x' + to_string(input_width) + ".txt"; // output from layer 2
    const string weight_filepath = "weights/layer4_" + to_string(output_channels) + 'x' + to_string(input_channels) + 'x' + to_string(kernel_size) + 'x' + to_string(kernel_size) + ".txt";
    const string bias_filepath = "biases/layer4_" + to_string(output_channels) + ".txt";


    vector<vector<float>> input = numpy_text_to_batch(batch_size,input_channels,input_height,input_width,batch_filepath);
    vector<float> weights = numpy_text_to_weight(output_channels,input_channels,kernel_size,weight_filepath);
    vector<float> biases = numpy_text_to_bias(output_channels,bias_filepath);


    ConvNet model(input_channels,output_channels,kernel_size,padding,stride);

    model.set_weights(weights);
    model.set_biases(biases);



    sycl::queue q(sycl::gpu_selector_v);
    model.device(q);

    vector<vector<float>> output = model.forward(input,batch_size,input_height,input_width);

    string output_filename = "csv/layer4_" + to_string(input_channels) + 'x' + to_string(output_channels) + ".csv";
    write_performance_time_to_csv(model.naive_time,model.reorder_time,model.direct_time,output_filename);

    for (int i=0;i< output[0].size();i++){
        cout << output[0][i] << " ";
    }

    /* (Compare result with the output from the pytorch Conv2d)

    string output_filepath = "output/layer2_output_32x64x32x32.txt"; // output

    vector<vector<vector<vector<float>>>> output_pytorch = numpy_text_to_batch(batch_size,output_channels,input_height,input_width,output_filepath);

    for (int batch=0;batch < batch_size;batch++){
        model.cmp_mat(output_pytorch[batch],output[batch]);
    }

    */
 
    return 0;
}