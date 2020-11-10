//compile : nvcc -o cuda_v2 cuda_v2.cu --expt-relaxed-constexpr
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
#include <cuComplex.h>
#include <thrust/complex.h>
#include <cuda_fp16.h>

using namespace std;

void print_weights(int rows, int cols, float* array){
  for(int i = 0; i< rows; i++){
      for(int j=0; j< cols ; j++){
        printf("%f " , array[i* cols + j]);
      }
    printf("\n");
  }
}

void print_hits(int rows, int cols, float* array){
  for(int i = 0; i< rows; i++){
      for(int j=0; j< cols ; j++){
        printf("%d " , int(array[i* cols + j]));
      }
    printf("\n");
  }
}

__global__ void kernel1(float *d_traindate, float *d_weight_matrix, int datapoint, float *d_radius,float *d_lrate, float *d_distances,int *d_bmu, int no_of_training_data, int no_of_features, int no_of_neurons){
    int tid = blockDim.x* blockIdx.x + threadIdx.x;
    float distance=0.0;
    for(int i=0; i< no_of_features; i++){
        distance+= abs (d_traindate[datapoint* no_of_features +i] - d_weight_matrix[tid * no_of_features + i ]);
    }

    d_distances[tid]=distance;
    __syncthreads();
    if(tid == 0){
      int temp_min_dis=100000;
      int temp_bmu=0;
      for(int i=0;i<no_of_neurons;i++){
        if(d_distances[i]<temp_min_dis){
          temp_min_dis = d_distances[i];
          temp_bmu = i;
        }
      }
      *d_bmu = temp_bmu;
      //printf(" cuda bmu = %d", *d_bmu);

    }
    //printf("%f ", distance);
    //*d_lrate=0.3;
}

__global__ void kernel2(float *d_traindate, float *d_weight_matrix, int *d_bmu,  int no_of_features, float *d_lrate, float *d_radius, int som_dim, int row){
    int tid = blockDim.x* blockIdx.x + threadIdx.x;
    int bmu=*d_bmu;
    float lrate = *d_lrate;
    float radius = *d_radius;

    //printf("bmu %d lrate %f radium %f", bmu, lrate , radius );
    int winR=bmu/som_dim;
    int winC=(bmu%som_dim);

    int currR=tid/som_dim;
    int currC=(tid%som_dim);
    float alpha = exp(- sqrt( ((winR-currR)*(winR-currR)) + ((winC-currC)*(winC-currC)) ) / (2*radius*radius));
    for(int j=0; j<no_of_features ; j++){
        d_weight_matrix[tid* no_of_features+ j]=d_weight_matrix[tid* no_of_features+ j] + ( lrate* alpha *( d_traindate[row* no_of_features + j]-d_weight_matrix[tid* no_of_features+ j]));
    }
}


__global__ void kernel4(float *d_traindate, float *d_traindate_class, float *d_hitmap,  float *d_weight_matrix, int datapoint,  float *d_distances,int *d_bmu, int no_of_features, int no_of_neurons, int no_of_classes, int label){

      int tid = blockDim.x* blockIdx.x + threadIdx.x;
      //
      if(tid< no_of_neurons){
        //printf(",%d",tid);
        float distance=0.0;
        for(int i=0; i< no_of_features; i++){
            distance+= abs (d_traindate[datapoint* no_of_features +i] - d_weight_matrix[tid * no_of_features + i ]);
        }

        d_distances[tid]=distance;
        __syncthreads();
        if(tid == 0){

            int temp_min_dis=100000;
            int temp_bmu=0;
            for(int i=0;i<no_of_neurons;i++){
              if(d_distances[i]<temp_min_dis){
                temp_min_dis = d_distances[i];
                temp_bmu = i;
              }
            }
            d_hitmap[temp_bmu  * no_of_classes + label] = d_hitmap[temp_bmu  * no_of_classes + label] +1;
            
        }

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
    float *hitmap;
    float *class_list; // assigned class for each neuron
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

    float *traindata_class= (float*)malloc(no_of_training_data*1 * sizeof(float));
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
    som_dim = 10; //number of neuron in one dimention // 25,10(e)
    neu_dim = no_of_features; // input chanels per neuron,kernel size
    radius[0]=som_dim/2.0;
    int no_of_neurons = som_dim*som_dim;
    weight_matrix = (float*)malloc( no_of_neurons * neu_dim * sizeof(float));//number of neurons
    class_list = (float*)malloc(som_dim*som_dim * sizeof(float));
    hitmap = (float*)malloc(som_dim * som_dim * no_of_classes * sizeof(float));//number of neurons
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
          hitmap[i  * no_of_classes + j] = 0.0;// initializing to zero hits
        }
    }

    //device variables

    float *d_lrate,*d_radius, *d_weight_matrix, *d_distances, *d_traindate;
    float *d_class_list, *d_hit_map, *d_traindata_class;
    int *d_bmu;
    cudaMalloc(&d_lrate, 1 * sizeof(float));
    cudaMalloc(&d_radius, 1 * sizeof(float));
    cudaMalloc(&d_bmu, 1 * sizeof(int));
    cudaMalloc(&d_weight_matrix, no_of_neurons * no_of_features * sizeof(float));
    cudaMalloc(&d_distances, no_of_neurons * sizeof(float));
    cudaMalloc(&d_class_list, no_of_neurons*  sizeof(float));
    cudaMalloc(&d_hit_map, no_of_neurons* no_of_classes* sizeof(float));
    cudaMalloc(&d_traindate, no_of_training_data* no_of_features * sizeof(float));
    cudaMalloc(&d_traindata_class, no_of_training_data*1* sizeof(float));



    cudaMemcpy(d_traindate, traindata, no_of_training_data* no_of_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_traindata_class, traindata_class, no_of_training_data*1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix, weight_matrix, no_of_neurons* no_of_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, h_distance_list, no_of_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_list, traindata_class, no_of_training_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hit_map, hitmap, no_of_neurons* no_of_classes * sizeof(float), cudaMemcpyHostToDevice);
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
    int grid_dim_k1 = (no_of_neurons + threads_per_block_k1 - 1) / threads_per_block_k1;


    printf("no of neurons %d\n", no_of_neurons);
    for (int epoch=0; epoch<epoches; epoch++)//epoches
    {
        cout << " epoch" << epoch << endl;
        lrate[0]= 0.49*(1-(epoch/epoches))+0.01;
        cudaMemcpy(d_lrate, lrate, 1 * sizeof(float), cudaMemcpyHostToDevice);

        for(int row = 0; row < no_of_training_data; row++)//no_of_training_data
        {

            //run a kernel to calculate distance to evey neu
            cudaEventRecord(start);
            kernel1<<<grid_dim_k1, threads_per_block_k1>>>(d_traindate, d_weight_matrix, row,d_radius,d_lrate, d_distances,d_bmu, no_of_training_data, no_of_features, no_of_neurons);
            cudaMemcpy(h_distance_list, d_distances, no_of_neurons * sizeof(float), cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            /*
            int temp_min_dis=100000;
            int temp_bmu=0;
            for(int i=0;i<no_of_neurons;i++){
              if(h_distance_list[i]<temp_min_dis){
                temp_min_dis = h_distance_list[i];
                temp_bmu = i;
              }
            }
            int *h_bmu=(int*)malloc(sizeof(int));
            h_bmu[0]=temp_bmu;
            printf("\n BMU for data record %d is %d",row, temp_bmu);
            cudaMemcpy(d_bmu, h_bmu, 1 * sizeof(int), cudaMemcpyHostToDevice);
            */
            ///cudaEventSynchronize
            //mem copy of d_distances
            //find bmu in cpu PRINT TO COMPARE WITH CPU
            //or atomic min
            //run a kernel to update weight mat
            cudaEventRecord(start);
            kernel2<<<grid_dim_k1, threads_per_block_k1>>>(d_traindate, d_weight_matrix, d_bmu,  no_of_features, d_lrate, d_radius, som_dim, row);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);



        }

        if(radius[0]/2.0 <=0)
          radius[0] = 1;
        else
          radius[0] = radius[0]/2.0;
        cudaMemcpy(d_radius, radius,1 * sizeof(float), cudaMemcpyHostToDevice);
    }

    clock_t end = clock();
  	double elapsed_secs_training = double(end - begin) / CLOCKS_PER_SEC;

    cudaMemcpy(weight_matrix, d_weight_matrix, no_of_neurons* no_of_features * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_traindata_class, traindata_class, no_of_training_data*1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hit_map, hitmap, no_of_neurons* no_of_classes * sizeof(float), cudaMemcpyHostToDevice);
    begin = clock();
    /*
    for(int row = 0; row < no_of_training_data; row++)//no_of_training_data
    {
              //for a  given data record, find the L1 distance to each neuron in SOM
              float min_dis=100000.0;
              int min_idx=-1;
              float distance=0;
              for(int neu=0; neu<no_of_neurons; neu++){
                  distance=0.0;
                  for(int col=0; col< no_of_features; col++){
                      distance += abs( traindata[row* no_of_features + col] - weight_matrix[ neu * no_of_features + col]) ;
                  }
                  if(distance<=min_dis){
                      min_dis = distance;
                      min_idx = neu;
                  }
              }
              hitmap[min_idx  * no_of_classes + traindata_class[row]] = hitmap[min_idx  * no_of_classes + traindata_class[row]] +1;
    }

    */

    //printf("grid_dim_k1 %d threads_per_block_k1 %d",grid_dim_k1, threads_per_block_k1);
    for(int row = 0; row < no_of_training_data; row++)//no_of_training_data
    {
              //for a  given data record, find the L1 distance to each neuron in SOM
              //printf("For data record: %d ", row);
              cudaEventRecord(start);
              kernel4<<<grid_dim_k1, threads_per_block_k1>>>(d_traindate, d_traindata_class, d_hit_map, d_weight_matrix, row,d_distances,d_bmu,  no_of_features, no_of_neurons, no_of_classes, traindata_class[row]);
              cudaEventRecord(stop);
              cudaEventSynchronize(stop);
              //printf("\n");

    }
    cudaMemcpy(hitmap, d_hit_map, no_of_neurons* no_of_classes * sizeof(int), cudaMemcpyDeviceToHost);


    end = clock();
  	double elapsed_secs_calculating_hitmap = double(end - begin) / CLOCKS_PER_SEC;

    cout << endl;
    //print_hits(som_dim*som_dim, no_of_classes, hitmap);

    begin = clock();
    //started = std::chrono::high_resolution_clock::now();

    int max=0;
    int max_idx=0;
    int tot=0;
    for(int neu=0; neu<no_of_neurons; neu++){
        max=0;max_idx=0;
        for(int col=0; col< no_of_classes; col++){
            if(hitmap[neu  * no_of_classes + col] >= max){
                max = hitmap[neu  * no_of_classes + col];
                max_idx = col;
            }

        }
        class_list[neu] = max_idx;
        //printf("class %d \n", class_list[neu]);
        tot+=max;
    }

    end = clock();
  	double elapsed_secs_calculating_class_assignment = double(end - begin) / CLOCKS_PER_SEC;


    printf("No of training data %d: train acc: %f \n", no_of_training_data, tot/float(no_of_training_data));

    //calculate test test_accuracy

    std::ifstream file3("testdata.csv");

    for(int row = 0; row < no_of_training_data; row++)
    {
        std::string line;
        std::getline(file3, line);
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
    std::ifstream file4("testdata_class.csv");
    for(int row = 0; row < no_of_training_data; ++row)
    {
          std::string line;
          std::getline(file4, line);
          if ( !file.good() )
              break;
          std::stringstream iss(line);
          std::string val;
          std::getline(iss, val, ',');
          std::stringstream convertor(val);
          convertor >> traindata_class[row];
    }

    begin = clock();
    //started = std::chrono::high_resolution_clock::now();

    int correct_count=0;
    for(int row = 0; row < no_of_training_data; row++)//no_of_training_data
    {
        //for a  given data record, find the L1 distance to each neuron in SOM
        float min_dis=100000.0;
        int min_idx=-1;
        float distance=0;
        for(int neu=0; neu<no_of_neurons; neu++){
            distance=0.0;
            for(int col=0; col< no_of_features; col++){
                distance += abs( traindata[row* no_of_features + col] - weight_matrix[ neu * no_of_features + col]) ;
            }
            if(distance<=min_dis){
                min_dis = distance;
                min_idx = neu;
            }
        }
        if(traindata_class[row]== class_list[min_idx]){
            correct_count++;
        }

    }
    end = clock();
  	double elapsed_secs_testing_accuracy_calc = double(end - begin) / CLOCKS_PER_SEC;

    cout << endl;

    printf("No of testing data %d: train acc: %f \n", no_of_training_data, correct_count/float(no_of_training_data));
    cout << "\nTraining time in Seconds (claculation weight matrix and fill the hitmap) : " << elapsed_secs_training <<endl;
    cout << "\nHitmap Calculation time in Seconds : " << elapsed_secs_calculating_hitmap <<endl;
    cout << "\nClass assignment calculation time in Seconds (find majority voted class and trin accuracy using hitmap) : " << elapsed_secs_calculating_class_assignment <<endl;
    cout << "\nTest accuracy calculation time in Seconds : " << elapsed_secs_testing_accuracy_calc <<endl;


    free(traindata);


    return 0;
}
