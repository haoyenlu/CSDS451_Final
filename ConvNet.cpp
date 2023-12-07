#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sycl/sycl.hpp>


#define COMP(a,b) (abs(a-b) < 0.001)

using namespace std;


class ConvNet{
    private:
        vector<vector<vector<vector<float>>>> weights; // shape: (output_channel,input_channel,kernel_size,kernel_size)
        vector<float> biases; // shape: output_channel
        unsigned input_channels;
        unsigned output_channels;
        unsigned stride;
        unsigned kernel;
        unsigned padding;

        sycl::queue Q;
    public:
        ConvNet(){}
        ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel, unsigned padding, unsigned stride);
        vector<vector<vector<vector<float>>>> forward(vector<vector<vector<vector<float>>>> input);

        void set_weights(vector<vector<vector<vector<float>>>> weights);
        void set_biases(vector<float> biases);

        vector<vector<vector<float>>> add_padding(vector<vector<vector<float>>> image);

        vector<vector<vector<float>>> im2col(vector<vector<vector<float>>> image);
        bool cmp_mat(vector<vector<vector<float>>> o1, vector<vector<vector<float>>> o2);

        // algorithm
        vector<vector<vector<float>>> reorder(vector<vector<vector<float>>> image,int input_height,int input_width,int output_height, int output_width);
        vector<vector<vector<float>>> naive(vector<vector<vector<float>>> image,int input_height,int input_width,int output_height,int output_width);
        vector<vector<vector<float>>> direct(vector<vector<vector<float>>> image, int input_height,int input_width, int output_height, int output_width);

        void device(sycl::queue &Q);
};


ConvNet::ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel,unsigned padding, unsigned stride){
    this->input_channels = input_channel;
    this->output_channels = output_channel;
    this->kernel = kernel;
    this->stride = stride;
    this->padding = padding;

    this->weights = vector<vector<vector<vector<float>>>>(output_channel,vector<vector<vector<float>>>(input_channel, vector<vector<float>>(kernel,vector<float> (kernel,0.5))));
    this->biases = vector<float>(output_channel,0);
}

void ConvNet::set_weights(vector<vector<vector<vector<float>>>> weights){
    for (int i=0;i<this->output_channels;i++){
        for (int j=0;j<this->input_channels;j++){
            for (int k=0;i<kernel;i++){
                for (int l=0;j<kernel;j++){
                    float w =  weights[i][j][k][l];
                    this->weights[i][j][k][l] = w;
                }
            }
        }
    }

}

void ConvNet::set_biases(vector<float> biases){
    for (int i=0;i<output_channels;i++){
        this->biases[i] = biases[i];       
    }
}

vector<vector<vector<float>>> ConvNet::add_padding(vector<vector<vector<float>>> input){
    size_t input_channels = input.size();
    size_t height = input[0].size(); // number of row
    size_t width = input[0][0].size(); // number of column

    // printf("Input Shape:(%zu,%zu,%zu)\n",input_channels,height,width);

    vector<vector<vector<float>>> input_with_padding(input_channels,vector<vector<float>> (height + this->padding * 2, vector<float> (width + this->padding * 2,0)));

    for (int i=0;i<input_channels;i++){
        for (int j=0;j<height;j++){
            for (int k=0;k<width;k++){
                input_with_padding[i][j+this->padding][k+this->padding] = input[i][j][k];
            }
        }
    }

    printf("Input with padding Shape:(%zu,%zu,%zu)\n",input_with_padding.size(),input_with_padding[0].size(),input_with_padding[0][0].size());

    return input_with_padding;

}

vector<vector<vector<vector<float>>>> ConvNet::forward(vector<vector<vector<vector<float>>>> input){
    int batches = input.size();
    int input_channel = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();

    int output_height = ((height - this->kernel + 2 * this->padding) / this->stride) + 1;
    int output_width = ((width - this->kernel + 2 * this->padding) / this->stride) + 1;
    vector<vector<vector<vector<float>>>> output_batch (batches, vector<vector<vector<float>>>(this->output_channels, vector<vector<float>> (output_height, vector<float>(output_width,0))));

    if (input_channel != this->input_channels){
        printf("Error: Input has %d channels, which is different than the required %d channels",input_channel,this->input_channels);
        return output_batch;
    }

    for (int image = 0; image < batches; image ++){
        // add padding 
        vector<vector<vector<float>>> image_with_padding = this->add_padding(input[image]);
        // run algorithm
        vector<vector<vector<float>>> naive_output = this->naive(image_with_padding,height,width,output_height,output_width);
        vector<vector<vector<float>>> reorder_output = this->reorder(image_with_padding,height,width,output_height,output_width);
        vector<vector<vector<float>>> direct_output = this->direct(image_with_padding,height,width,output_height,output_width);
        // compare result
        cout << "------------Comparing naive and reorder-------------------\n";
        this->cmp_mat(naive_output,reorder_output);
        cout << "-------------Comparing naive and direct------------------\n";
        this->cmp_mat(naive_output,direct_output);
        
        output_batch[image] = naive_output;
    }
    return output_batch;
}

// naive convolution algorithm
vector<vector<vector<float>>> ConvNet::naive(vector<vector<vector<float>>> image,int input_height,int input_width,int output_height,int output_width){
    vector<vector<vector<float>>> output (this->output_channels,vector<vector<float>> (output_height, vector<float> (output_width,0)));

    for (int i=0;i<this->input_channels;i++){
        for (int j=0;j<this->output_channels;j++){
            for (int k=0;k<output_width;k++){
                for (int l=0;l<output_height;l++){
                    for (int m=0;m<this->kernel;m++){
                        for(int n=0;n<this->kernel;n++){
                            int input_row = l * this->stride + n;
                            int input_col = k * this->stride + m;
                            if (input_row >= input_height || input_col >= input_width) continue;
                            else output[j][l][k] += image[i][input_row][input_col] * this->weights[j][i][n][m];             
                        }
                    }
                    output[j][l][k] += this->biases[j];
                }
            }
        }
    }
    return output;
}


// reorder the loop of naive convolution algorithm
vector<vector<vector<float>>> ConvNet::reorder(vector<vector<vector<float>>> image, int input_height, int input_width, int output_height, int output_width){
    vector<vector<vector<float>>> output (this->output_channels,vector<vector<float>> (output_height, vector<float> (output_width,0)));

    for (int l=0;l<output_height;l++){ // H_o ( output height )
        for (int n=0;n<this->kernel;n++){ // H_f ( kernel height )
            for (int m=0;m<this->kernel;m++){ // W_f ( kernel width )
                for (int i=0;i<this->input_channels;i++){ //C_i ( input channel )
                    for (int k=0;k<output_width;k++){ // W_o ( output width )
                        for (int j=0;j<this->output_channels;j++){ // C_o ( output channel )
                            int input_row = l * this->stride + n;
                            int input_col = k * this->stride + m;
                            if (input_row >= input_height || input_col >= input_width) continue;
                            else {
                                output[j][l][k] += image[i][input_row][input_col] * this->weights[j][i][n][m];   
                                output[j][l][k] += this->biases[j];
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}


vector<vector<vector<float>>> ConvNet::direct(vector<vector<vector<float>>> input, int input_height,int input_width, int output_height,int output_width){
    cout << "-----------------------"<< "Run direct algorithm" << "-----------------------------------" <<"\n";
    sycl::queue q(sycl::default_selector_v);
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    
    float **output_host = new float*[this->output_channels];
    for (int i=0;i<this->output_channels;i++){
        output_host[i] = new float[output_height * output_width];
        for (int j=0;j<output_height;j++){
            for (int k=0;k<output_width;k++){
                output_host[i][j*output_width + k] = 0;
            }
        }
    }

    float **input_host = new float*[this->input_channels];
    for (int i=0;i<this->input_channels;i++){
        input_host[i] = new float[input_height * input_width];
        for (int h=0;h<input_height;h++){
            for (int w=0;w<input_width;w++){
                input_host[i][h*input_width + w] = input[i][h][w];
            }
        }
    }

    float **weight_host = new float*[this->input_channels * this->output_channels];
    for (int i=0;i<this->output_channels;i++){
        for (int j=0;j<this->input_channels;j++){   
            weight_host[i*this->input_channels + j] = new float[this->kernel*this->kernel];
            for (int k=0;k<this->kernel;k++){
                for (int q=0;q<this->kernel;q++){
                    weight_host[i*this->input_channels + j][k * this->kernel + q] = this->weights[i][j][k][q];
                }
            }
        }
    }


    float *bias_host = new float[this->output_channels];
    for (int i=0;i<this->output_channels;i++){
        bias_host[i] = this->biases[i];
    }

    sycl::range<2> output_range{static_cast<size_t>(this->output_channels),static_cast<size_t>(output_height * output_width)};
    sycl::range<2> input_range{static_cast<size_t>(this->input_channels),static_cast<size_t>(input_height * input_width)};
    sycl::range<2> weight_range{static_cast<size_t>(this->output_channels * this->input_channels), static_cast<size_t>(this->kernel * this->kernel)};
    sycl::range<1> bias_range{static_cast<size_t>(this->output_channels)};

    sycl::buffer<float,2> output_buf(*output_host,output_range);
    sycl::buffer<float,2> input_buf(*input_host,input_range);
    sycl::buffer<float,2> weight_buf(*weight_host,weight_range);
    sycl::buffer<float,1> bias_buf(bias_host,bias_range);
    

    this->Q.submit([&](sycl::handler &h){
        sycl::accessor output(output_buf,h);
        sycl::accessor input(input_buf,h);
        sycl::accessor weights(weight_buf,h);
        sycl::accessor biases(bias_buf,h);

        int output_channels = this->output_channels;
        int input_channels = this->input_channels;
        int stride = this->stride;
        int kernel = this->kernel;

        int output_chn_block_size = 16;
        int input_chn_block_size = 1;
        int output_width_block_size = 8;

        int output_chn_block_cnt = static_cast<int>(output_channels / output_chn_block_size );
        int input_chn_block_cnt = static_cast<int>(input_channels / input_chn_block_size);
        int output_width_block_cnt = static_cast<int>(output_width / output_width_block_size);


        h.parallel_for(output_chn_block_cnt,[=](auto j){
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
                                        int block_input_width = stride * k * output_width_block_size + kk + m;
                                        if (block_output_width >= output_width || block_input_width >= input_width) break;

                                        for (int jj=0;jj<output_chn_block_size;jj++){
                                            int block_output_chn = j * output_chn_block_size  + jj; 
                                            if (block_output_chn >= output_channels) break;
                                            float ans = input[block_input_chn][block_input_height * input_width + block_input_width] * weights[block_output_chn * input_channels + block_input_chn][n* kernel + m];

                                            output[block_output_chn][l * output_width + block_output_width] += ans;
                                            output[block_output_chn][l * output_width + block_output_width] += biases[block_output_chn];
                                        }       
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    this->Q.wait();
    });

    cout << "--End Parallel\n";
    sycl::host_accessor<float,2> output_result(output_buf,output_range);

    vector<vector<vector<float>>> output (this->output_channels, vector<vector<float>> (output_height, vector<float>(output_width,0)));
    for (int i=0;i<this->output_channels;i++){
        for(int j=0;j<output_height;j++){
            for (int k=0;k<output_width;k++){
                output[i][j][k] = output_result[i][j*output_width + k];
            }
        }
    }

    delete[] output_host;
    delete[] input_host;
    delete[] weight_host;
    delete[] bias_host;
    return output;
}

bool ConvNet::cmp_mat(vector<vector<vector<float>>> o1, vector<vector<vector<float>>> o2){
    // compare size
    int o1_c = o1.size();
    int o1_h = o1[0].size();
    int o1_w = o1[0][0].size();

    int o2_c = o2.size();
    int o2_h = o2[0].size();
    int o2_w = o2[0][0].size();
    if (o1_c != o2_c || o1_h != o2_h || o1_w != o2_w ){
        printf("Error: Two matrix has different size. matrix_1 size: (%d,%d,%d), matrix_2 size:(%d,%d,%d)\n", o1_c,o1_h,o1_w,o2_c,o2_h,o2_w);
        return false;
    }
    else{
        printf("Matrix Size: (%d,%d,%d)\n",o1_c,o1_h,o1_w);
    }

    // compare value
    bool same = true;
    int number_of_difference = 0;
    for (int c=0;c<o1_c;c++){
        for (int h=0;h<o1_h;h++){
            for (int w=0;w<o1_w;w++){
                if (!COMP(o1[c][h][w],o2[c][h][w])){
                    //printf("Two matrix has different value at (%d,%d,%d). matrix_1 value: %f, matrix_2 value: %f\n",c,h,w,o1[c][h][w], o2[c][h][w]);
                    number_of_difference += 1;
                    same = false;
                }
            }
        }
    }
    cout << "Number of difference: "<<number_of_difference << endl;
    return same;
}

void ConvNet::device(sycl::queue &Q){
    this->Q = Q;
    cout << "Device: " << this->Q.get_device().get_info<sycl::info::device::name>() << "\n";
}


vector<vector<vector<vector<float>>>> numpy_text_to_batch(int batch_size, int input_channels, int height, int width, string filepath){
    vector<vector<vector<vector<float>>>> batch;

    vector<vector<float>> data(batch_size * input_channels,vector<float>(height * width,0));

    ifstream file;
    file.open(filepath,ios::in);

    if (!file.is_open()){
        cout<< "Error: Cannot open file!" << endl;
        return batch;
    }
    
    for (int i=0;i<batch_size * input_channels;i++){
        for (int j=0;j<height * width;j++){
            file >> data[i][j];
            if ( ! file ) {
                cout << "Error reading file for element " << i << "," << j << endl; 
            }
        }
    }

    // printf("Data Shape:(%d,%d)",data.size(),data[0].size());

    batch = vector<vector<vector<vector<float>>>>(batch_size,vector<vector<vector<float>>>(input_channels,vector<vector<float>>(height,vector<float>(width,0))));
    
    for (int i=0;i<batch_size;i++){
        for (int j=0;j<input_channels;j++){
            for (int k=0;k<height;k++){
                for (int l=0;l<width;l++){
                    batch[i][j][k][l] = data[i * 3 + j][k*32 + l];
                }
            }
        }
    }

    return batch;
}



int main(){
    const string filepath = "image_0.txt";

    vector<vector<vector<vector<float>>>> images = numpy_text_to_batch(32,3,32,32,filepath);

    ConvNet model(3,64,3,1,1);

    sycl::queue q(sycl::gpu_selector_v);
    model.device(q);

    cout << "Input shape:(" << images.size() << "," << images[0].size() << "," << images[0][0].size() << "," << images[0][0][0].size() <<   ")" << endl;

    vector<vector<vector<vector<float>>>> output = model.forward(images);

    cout << "Output shape:(" << output.size() << "," << output[0].size() << "," << output[0][0].size() << "," << output[0][0][0].size() << ")\n";


 
    return 0;
}