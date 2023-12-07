#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sycl/sycl.hpp>

#include "dpc_common.hpp"

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

        float****weights_usm;
        float*biases_usm;

    public:
        ConvNet(){};
        ~ConvNet();
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
        vector<vector<vector<float>>> direct_without_parallel(vector<vector<vector<float>>> image, int input_height,int input_width, int output_height, int output_width);

        void device(sycl::queue Q);
};


ConvNet::ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel,unsigned padding, unsigned stride){
    this->input_channels = input_channel;
    this->output_channels = output_channel;
    this->kernel = kernel;
    this->stride = stride;
    this->padding = padding;

    this->weights = vector<vector<vector<vector<float>>>>(output_channel,vector<vector<vector<float>>>(input_channel, vector<vector<float>>(kernel,vector<float> (kernel,0.5))));
    this->biases = vector<float>(output_channel,0.5);
}

ConvNet::~ConvNet(){
    free(this->weights_usm,this->Q);
    free(this->biases_usm,this->Q);
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

    height = height + 2 * this->padding;
    width = width + 2 * this->padding;

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
                            output[j][l][k] += image[i][input_row][input_col] * this->weights[j][i][n][m];         
                            output[j][l][k] += this->biases[j];    
                        }
                    }
                }
            }
        }
    }

    
    double time_elapsed = time.Elapsed();
    cout << "Naive Convolution Time: " << time_elapsed << endl;

    return output;
}


// reorder the loop of naive convolution algorithm
vector<vector<vector<float>>> ConvNet::reorder(vector<vector<vector<float>>> image, int input_height, int input_width, int output_height, int output_width){
    vector<vector<vector<float>>> output (this->output_channels,vector<vector<float>> (output_height, vector<float> (output_width,0)));
    
    dpc_common::TimeInterval time; // Timing the Convolution Algorithm

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
    
    double time_elapsed = time.Elapsed();
    cout << "Reorder Convolution Time: " << time_elapsed << endl;

    return output;
}


vector<vector<vector<float>>> ConvNet::direct(vector<vector<vector<float>>> input, int input_height,int input_width, int output_height,int output_width){
    cout << "-----------------------"<< "Run direct algorithm" << "-----------------------------------" <<"\n";

    //sycl::queue q(sycl::gpu_selector_v);
    //std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    float ***output_share = sycl::malloc_shared<float**>(this->output_channels,this->Q);
    for (int i=0;i<this->output_channels;i++){
        output_share[i] = sycl::malloc_shared<float*>(output_height, this->Q);
        for (int j=0;j<output_height;j++){
            output_share[i][j] = sycl::malloc_shared<float>(output_width,this->Q);
            for (int k=0;k<output_width;k++){
                output_share[i][j][k] = 0;
            }
        }
    }
    float ***image_share = sycl::malloc_shared<float**>(this->input_channels,this->Q);
    for (int i=0;i<this->input_channels;i++){
        image_share[i] = sycl::malloc_shared<float*>(input_height,this->Q);
        for (int h=0;h<input_height;h++){
            image_share[i][h] = sycl::malloc_shared<float>(input_width,this->Q);
            for (int w=0;w<input_width;w++){
                image_share[i][h][w] = input[i][h][w];
            }
        }
    }
    
    float ****weights_share = this->weights_usm;
    float *biases_share = this->biases_usm;

    dpc_common::TimeInterval time; // Timing the Convolution Algorithm

    this->Q.submit([&](sycl::handler &h){
        int output_channels = this->output_channels;
        int input_channels = this->input_channels;
        int stride = this->stride;
        int kernel = this->kernel;

        int output_chn_block_size = 1;
        int input_chn_block_size = 1;
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
                                            float ans = image_share[block_input_chn][block_input_height][block_input_width] * weights_share[block_output_chn][block_input_chn][n][m];

                                            output_share[block_output_chn][l][block_output_width] += ans;
                                            output_share[block_output_chn][l][block_output_width] += biases_share[block_output_chn];
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

    vector<vector<vector<float>>> output_result (this->output_channels, vector<vector<float>> (output_height, vector<float>(output_width,0)));
    for (int i=0;i<this->output_channels;i++){
        for(int j=0;j<output_height;j++){
            for (int k=0;k<output_width;k++){
                output_result[i][j][k] = output_share[i][j][k];
            }
        }
    }

    free(output_share,this->Q);
    free(image_share,this->Q);

    return output_result;
}

vector<vector<vector<float>>> ConvNet::direct_without_parallel(vector<vector<vector<float>>> input, int input_height,int input_width, int output_height,int output_width){
    vector<vector<vector<float>>> output (this->output_channels,vector<vector<float>> (output_height, vector<float> (output_width,0)));

    int output_chn_block_size = 16;
    int input_chn_block_size = 1;
    int output_width_block_size = 8;

    int output_chn_block_cnt = static_cast<int>(this->output_channels / output_chn_block_size );
    int input_chn_block_cnt = static_cast<int>(this->input_channels / input_chn_block_size);
    int output_width_block_cnt = static_cast<int>(output_width / output_width_block_size);

    for (int j=0;j<output_chn_block_cnt;j++){
        for (int i=0;i< input_chn_block_cnt;i++){
            for (int l=0;l<output_height;l++){
                for (int k=0;k<output_width_block_cnt;k++){
                    for (int n=0;n<this->kernel;n++){
                        int block_input_height = l * stride + n;
                        if (block_input_height >= input_height) break;

                
                        for (int m=0;m<this->kernel;m++){
                            for (int ii=0;ii<input_chn_block_size;ii++){
                                int block_input_chn = i * input_chn_block_size + ii;
                                if (block_input_chn >= input_channels) break;

                                for (int kk=0;kk<output_width_block_size;kk++){
                                    int block_output_width = k * output_width_block_size  + kk;
                                    int block_input_width = stride * block_output_width + m;
                                    if (block_output_width >= output_width || block_input_width >= input_width) break;

                                    for (int jj=0;jj<output_chn_block_size;jj++){
                                        int block_output_chn = j * output_chn_block_size  + jj; 
                                        if (block_output_chn >= output_channels) break;
                                        float ans = input[block_input_chn][block_input_height][block_input_width] * this->weights[block_output_chn][block_input_chn][n][m];

                                        output[block_output_chn][l][block_output_width] += ans;
                                        output[block_output_chn][l][block_output_width] += biases[block_output_chn];
                                    }       
                                }
                            }
                        }
                    }
                }
            }
        }
    }
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

    // compare value
    bool same = true;
    int number_of_difference = 0;
    for (int c=0;c<o1_c;c++){
        for (int h=0;h<o1_h;h++){
            for (int w=0;w<o1_w;w++){
                if (!COMP(o1[c][h][w],o2[c][h][w])){
                    if (same) printf("Two matrix has different value at (%d,%d,%d). matrix_1 value: %f, matrix_2 value: %f\n",c,h,w,o1[c][h][w], o2[c][h][w]);
                    number_of_difference += 1;
                    same = false;
                }
            }
        }
    }
    cout << "Number of difference: "<<number_of_difference << endl;
    return same;
}

void ConvNet::device(sycl::queue Q){
    this->Q = Q;
    cout << "Device: " << this->Q.get_device().get_info<sycl::info::device::name>() << "\n";

    this->weights_usm = sycl::malloc_shared<float***>(this->output_channels,this->Q);
    for (int i=0;i<this->output_channels;i++){
        weights_usm[i] = sycl::malloc_shared<float**>(this->input_channels,this->Q);
        for (int j=0;j<this->input_channels;j++){
            weights_usm[i][j] = sycl::malloc_shared<float*>(this->kernel,this->Q);
            for (int k=0;k<this->kernel;k++){
                weights_usm[i][j][k] = sycl::malloc_shared<float>(this->kernel,this->Q);
                for (int q=0;q<this->kernel;q++){
                    weights_usm[i][j][k][q] = this->weights[i][j][k][q];
                }
            }
        }
    }

    this->biases_usm = sycl::malloc_shared<float>(this->output_channels,this->Q);
    for (int i=0;i<this->output_channels;i++){
        biases_usm[i] = this->biases[i];
    }
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
    const string filepath = "layer2_batch_32x64x16x16.txt";

    int batch_size = 32;
    int input_channels = 64;
    int output_channels = 128;
    int input_height = 16;
    int input_width = 16;

    vector<vector<vector<vector<float>>>> images = numpy_text_to_batch(batch_size,input_channels,input_height,input_width,filepath);

    ConvNet model(input_channels,output_channels,3,1,1);

    sycl::queue q(sycl::gpu_selector_v);
    model.device(q);

    cout << "Input shape:(" << images.size() << "," << images[0].size() << "," << images[0][0].size() << "," << images[0][0][0].size() <<   ")" << endl;

    vector<vector<vector<vector<float>>>> output = model.forward(images);

    cout << "Output shape:(" << output.size() << "," << output[0].size() << "," << output[0][0].size() << "," << output[0][0][0].size() << ")\n";


 
    return 0;
}