#include <vector>
#include <iostream>
#include "npy.hpp"
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
        vector<vector<vector<float>>> forward(vector<vector<vector<float>>> input);
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

    printf("Input Shape:(%zu,%zu,%zu)\n",input_channels,height,width);

    vector<vector<vector<float>>> input_with_padding(input_channels,vector<vector<float>> (height + this->padding * 2, vector<float> (width + this->padding * 2,0)));

    for (int i=0;i<input_channels;i++){
        for (int j=0;j<height;j++){
            for (int k=0;k<width;k++){
                input_with_padding[i][j+this->padding][k+this->padding] = input[i][j][k];
            }
        }
    }

    return input_with_padding;

}

vector<vector<vector<float>>> ConvNet::forward(vector<vector<vector<float>>> input){
    int input_channel = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    int output_height = ((height - this->kernel + 2 * this->padding) / this->stride) + 1;
    int output_width = ((width - this->kernel + 2 * this->padding) / this->stride) + 1;
    vector<vector<vector<float>>> output (this->output_channels, vector<vector<float>> (output_height, vector<float>(output_width,0)));

    if (input_channel != this->input_channels){
        printf("Error: Input has %s channels, which is different than the required %s channels",input_channel,this->input_channels);
        return output;
    }

    // naive
    for (int i=0;i<input_channel;i++){
        for (int j=0;j<this->output_channels;j++){
            for (int k=0;k<output_height;k++){
                for (int l=0;l<output_width;l++){
                    for (int m=0;m<this->kernel && k * this->stride + m < output_width;m++){
                        for(int n=0;n<this->kernel && l * this->stride + n < output_height;n++){
                            output[j][k][l] += input[i][k*this->stride + m][l*this->stride + n] * this->weights[m][n];             
                        }
                    }
                }
            }
        }
    }


    return output;

}


vector<vector<float>> read_numpy_from_text(int rows, int cols, string filepath){
    vector<vector<float>> data(rows,vector<float>(cols,0));

    ifstream file;
    file.open(filepath,ios::in);

    if (!file.is_open()){
        cout<< "Error: Cannot open file!" << endl;
        return data;
    }
    
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            file >> data[i][j];
            if ( ! file ) {
                cout << "Error reading file for element " << i << "," << j << endl; 
            }
        }
    }

    return data;
}



int main(){
    const string filepath = "image_0.txt";

    vector<vector<float>> numpyArray = read_numpy_from_text(96,1024,filepath);

    cout << "Input Size:(" << numpyArray.size() << "," << numpyArray[0].size() << ")\n";

    vector<vector<vector<float>>> image (3,vector<vector<float>>(32, vector<float>(32)));

    for (int i=0;i<3;i++){
        for (int j=0;j<32;j++){
            for (int k=0;k<32;k++){
                image[i][j][k] = numpyArray[i][j*32 + k];
            }
        }
    }

    ConvNet model(3,32,3,1,1);

    vector<vector<vector<float>>> image_with_padding = model.add_padding(image);

    cout << "Image with padding shape:(" << image_with_padding.size() << "," << image_with_padding[0].size() << "," << image_with_padding[0][0].size() << ")" << endl;

    vector<vector<vector<float>>> output = model.forward(image_with_padding);

    cout << "Output shape:(" << output.size() << "," << output[0].size() << "," << output[0][0].size() << ")\n";
 
    return 0;
}