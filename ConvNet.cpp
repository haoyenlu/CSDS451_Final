#include <vector>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;


class ConvNet{
    private:
        vector<vector<float>> weights;
        unsigned input_channels;
        unsigned output_channels;
        unsigned stride;
        unsigned kernel;
        unsigned padding;

    public:
        ConvNet(){}
        ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel, unsigned padding, unsigned stride);
        vector<vector<vector<vector<float>>>> forward(vector<vector<vector<vector<float>>>> input);
        void set_weights(vector<vector<float>> weights);
        vector<vector<vector<float>>> add_padding(vector<vector<vector<float>>> input);
};


ConvNet::ConvNet(unsigned input_channel, unsigned output_channel, unsigned kernel,unsigned padding, unsigned stride){
    this->input_channels = input_channel;
    this->output_channels = output_channel;
    this->kernel = kernel;
    this->stride = stride;
    this->padding = padding;

    this->weights = vector<vector<float>>(kernel,vector<float> (kernel,0.5));
}

void ConvNet::set_weights(vector<vector<float>> weights){
    for (int i=0;i<kernel;i++){
        for (int j=0;j<kernel;j++){
            this->weights[i][j] = weights[i][j];
        }
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

    // printf("Input with padding Shape:(%zu,%zu,%zu)\n",input_with_padding.size(),input_with_padding[0].size(),input_with_padding[0][0].size());

    return input_with_padding;

}

vector<vector<vector<vector<float>>>> ConvNet::forward(vector<vector<vector<vector<float>>>> input){
    int batches = input.size();
    int input_channel = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();

    int output_height = ((height - this->kernel + 2 * this->padding) / this->stride) + 1;
    int output_width = ((width - this->kernel + 2 * this->padding) / this->stride) + 1;
    vector<vector<vector<vector<float>>>> output (batches, vector<vector<vector<float>>>(this->output_channels, vector<vector<float>> (output_height, vector<float>(output_width,0))));

    if (input_channel != this->input_channels){
        printf("Error: Input has %s channels, which is different than the required %s channels",input_channel,this->input_channels);
        return output;
    }

    // naive 

    for (int image = 0; image < batches; image ++){
        vector<vector<vector<float>>> image_with_padding = this->add_padding(input[image]);

        for (int i=0;i<input_channel;i++){
            for (int j=0;j<this->output_channels;j++){
                for (int k=0;k<output_height;k++){
                    for (int l=0;l<output_width;l++){
                        for (int m=0;m<this->kernel && k * this->stride + m < width;m++){
                            for(int n=0;n<this->kernel && l * this->stride + n < height;n++){
                                output[image][j][k][l] += image_with_padding[i][k*this->stride + m][l*this->stride + n] * this->weights[m][n];             
                            }
                        }
                    }
                }
            }
        }
    }



    return output;

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


    cout << "Input shape:(" << images.size() << "," << images[0].size() << "," << images[0][0].size() << "," << images[0][0][0].size() <<   ")" << endl;

    vector<vector<vector<vector<float>>>> output = model.forward(images);

    cout << "Output shape:(" << output.size() << "," << output[0].size() << "," << output[0][0].size() << "," << output[0][0][0].size() << ")\n";


 
    return 0;
}