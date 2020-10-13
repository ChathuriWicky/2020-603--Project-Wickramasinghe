//compile : nvcc -o cuda_v1 cuda_v1.cu
//run : ./cuda_v1
/*
test 1: epoches=5; som_dim=16; train_acc=72.2
test 1: epoches=10; som_dim=16; train_acc=76.2

*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>
#include <iostream>
#include <string>
#include <bitset>
#include <time.h>
#include <map>
#include <vector>
#include <set>
#include<list>
#include<random>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdint.h>
#include <iterator>
#include<algorithm>
#include <float.h>
#include <math.h>
#include <fcntl.h>
#include <string>
#include <bitset>
#include <time.h>



using namespace std;

void print_weights(int rows, int cols, float* array){
  for(int i = 0; i< rows; i++){
      for(int j=0; j< cols ; j++){
        printf("%f " , array[i* cols + j]);
      }
    printf("\n");
  }
}

void print_hits(int rows, int cols, int* array){
  for(int i = 0; i< rows; i++){
      for(int j=0; j< cols ; j++){
        printf("%d " , array[i* cols + j]);
      }
    printf("\n");
  }
}

int main(int argc, char *argv[]){

    // defining SOM
    int som_dim, neu_dim;// SOM network size, neuDim
    float *weight_matrix; //weight matrix
    float *lrate=(float*)malloc(1*sizeof(float));

    float *radius=(float*)malloc(1*sizeof(float));
    int epoches;
    int bmu;
    int *hitmap;
    int *class_list; // assigned class for each neuron
    int no_of_classes;
    float train_acc, test_accuracy;
    int no_of_training_data=2500;
    int img_dim=28;
    int no_of_features= img_dim * img_dim;
    int nums[no_of_training_data];


    cout << "SOM data loading no_of_training_data " << no_of_training_data << endl;
    cout << endl;

    std::ifstream file("traindata.csv");
    float *traindata= (float*) malloc (no_of_training_data * no_of_features * sizeof(float));
    for(int row = 0; row < no_of_training_data; row++)
    {
        std::string line;
        std::getline(file, line);
        if ( !file.good() ){
            break;
          }
        std::stringstream iss(line);
        for (int col = 0; col < no_of_features; col++)
        {
            std::string val;
            std::getline(iss, val, ',');
            if ( !iss.good() ){
              break;
            }
            std::stringstream convertor(val);
            convertor >> traindata[row * no_of_features + col];
        }
    }

    int *traindata_class= (int*)malloc(no_of_training_data * sizeof(int));
    std::ifstream file2("traindata_class.csv");
    int nn=0;
    for(int row = 0; row < no_of_training_data; ++row)
    {
          std::string line;
          std::getline(file2, line);
          if ( !file.good() )
              break;
          std::stringstream iss(line);
          std::string val;
          std::getline(iss, val, ',');
          std::stringstream convertor(val);
          convertor >> traindata_class[row];

          nums[nn]=nn;
          nn++;
    }

    cout << endl;

    //SOM initialization

    epoches=10;
    no_of_classes=10;
    lrate[0]=0.49;
    printf("l rate %f", lrate[0]);
    som_dim = 10; //number of neuron in one dimention // 25,10(e)
    neu_dim = no_of_features; // input chanels per neuron,kernel size
    radius[0]=som_dim/2.0;
    int no_of_neurons = som_dim*som_dim;
    weight_matrix = (float*)malloc( no_of_neurons * neu_dim * sizeof(float));//number of neurons
    class_list = (int*)malloc(som_dim*som_dim * sizeof(int));
    hitmap = (int*)malloc(som_dim * som_dim * no_of_classes * sizeof(int));//number of neurons
    float *h_distance_list=(float*)malloc(no_of_neurons * sizeof(float));

    for(int i = 0; i< som_dim*som_dim; i++){
          for(int j=0; j< no_of_features; j++){
            weight_matrix[i* neu_dim + j] = rand()/(float)RAND_MAX;//0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.0-0.0)));; // initializing to a random weight
          }
          class_list[i] =0;
          h_distance_list[i]=0.0;
    }

    for(int i = 0; i< som_dim*som_dim;i++){
        for(int j=0; j< no_of_classes ; j++){
          hitmap[i  * no_of_classes + j] = 0;// initializing to zero hits
        }
    }

    //device variables

    float *d_lrate,*d_radius, *d_weight_matrix, *d_distances, *d_traindate, *d_traindata_class;
    int *d_class_list, *d_hit_map;
    int *d_bmu;
    cudaMalloc(&d_lrate, 1 * sizeof(float));
    cudaMalloc(&d_radius, 1 * sizeof(float));
    cudaMalloc(&d_bmu, 1 * sizeof(int));
    cudaMalloc(&d_weight_matrix, no_of_neurons * no_of_features * sizeof(float));
    cudaMalloc(&d_distances, no_of_neurons * sizeof(float));
    cudaMalloc(&d_class_list, no_of_neurons*  sizeof(int));
    cudaMalloc(&d_hit_map, no_of_neurons* no_of_classes* sizeof(int));
    cudaMalloc(&d_traindate, no_of_training_data* no_of_features * sizeof(float));
    cudaMalloc(&d_traindata_class, no_of_training_data* sizeof(int));



    cudaMemcpy(d_traindate, traindata, no_of_training_data* no_of_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_traindata_class, traindata_class, no_of_training_data * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix, weight_matrix, no_of_neurons* no_of_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, h_distance_list, no_of_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_list, traindata_class, no_of_training_data * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hit_map, hitmap, no_of_neurons* no_of_classes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lrate, lrate, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_radius, radius,1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bmu, 0, 1 * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    clock_t begin = clock();
    auto started = std::chrono::high_resolution_clock::now();
    //read and store

    //print_weights(som_dim*som_dim, neu_dim, weight_matrix);

    //weight update using the traning trainData
    int threads_per_block_k1 = 32;
    int grid_dim_k1 = (no_of_training_data + threads_per_block_k1 - 1) / threads_per_block_k1;


    printf("no of neurons %d", no_of_neurons);
    for (int epoch=0; epoch<epoches; epoch++)//epoches
    {

        for(int row = 0; row < no_of_training_data; row++)//no_of_training_data
        {

            //run a kernel to calculate distance to evey neu
            cudaEventRecord(start);
            //kernel1<<<gridSize, blockSize>>>(d_dataset, d_distance_mat, no_of_data_records,no_of_features);
            //cudaMemcpy(h_distance_mat, d_distance_mat, no_of_data_records* no_of_data_records * sizeof(float), cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            ///cudaEventSynchronize
            //mem copy of d_distances
            //find bmu in cpu
            //or atomic min
            //run a kernel to update weight mat
            cudaEventRecord(start);
            //kernel2<<<gridSize, blockSize>>>(d_dataset, d_distance_mat, no_of_data_records,no_of_features);
            //cudaMemcpy(h_distance_mat, d_distance_mat, no_of_data_records* no_of_data_records * sizeof(float), cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            for(int neu=0; neu<no_of_neurons; neu++){

            }
            //int winR=min_idx/som_dim;
            ///int winC=(min_idx%som_dim);
            for(int i = 0; i< som_dim*som_dim ; i++){

            }
        }






    }

    clock_t end = clock();
  	double elapsed_secs_training = double(end - begin) / CLOCKS_PER_SEC;



    free(traindata);


    return 0;
}
