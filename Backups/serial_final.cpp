//compile : g++ -o serial.exe serial.cpp -std=c++11
//run : ./serial.exe
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
#include <algorithm>
#include <sstream>
#include <iomanip>


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
    float lrate;
    float radius;
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
    lrate=0.49;
    som_dim = 10; //number of neuron in one dimention // 25,10(e)
    neu_dim = no_of_features; // input chanels per neuron,kernel size
    radius=som_dim/2.0;
    int no_of_neurons = som_dim*som_dim;
    weight_matrix = (float*)malloc( no_of_neurons * neu_dim * sizeof(float));//number of neurons
    class_list = (int*)malloc(som_dim*som_dim * sizeof(int));
    hitmap = (int*)malloc(som_dim * som_dim * no_of_classes * sizeof(int));//number of neurons

    for(int i = 0; i< som_dim*som_dim; i++){
          for(int j=0; j< no_of_features; j++){
            weight_matrix[i* neu_dim + j] = rand()/(float)RAND_MAX;//0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.0-0.0)));; // initializing to a random weight
          }
          class_list[i] =0;
    }

    for(int i = 0; i< som_dim*som_dim;i++){
        for(int j=0; j< no_of_classes ; j++){
          hitmap[i  * no_of_classes + j] = 0;// initializing to zero hits
        }
    }

    clock_t begin = clock();
    auto started = std::chrono::high_resolution_clock::now();
    //read and store

    //print_weights(som_dim*som_dim, neu_dim, weight_matrix);

    //weight update using the traning trainData
    printf("no of neurons %d", no_of_neurons);
    for (int epoch=0; epoch<epoches; epoch++)//epoches
    {
        lrate = 0.49*(1-(epoch/epoches))+0.01;
        cout << " epoch" << epoch << endl;
        std::random_shuffle(nums, nums + no_of_training_data);
        for(int row = 0; row < no_of_training_data; row++)//no_of_training_data
        {
            int rid = nums [row];
            float *distances = (float*)malloc(no_of_neurons * sizeof(float));
            float min_dis=100000.0;
            int min_idx=-1;
            for(int neu=0; neu<no_of_neurons; neu++){
                distances[neu]=0.0;
                for(int col=0; col< no_of_features; col++){
                    distances[neu] += abs( traindata[rid* no_of_features + col] - weight_matrix[ neu * no_of_features + col]) ;
                }
                if(distances[neu]<=min_dis){
                    min_dis = distances[neu];
                    min_idx = neu;
                }
            }
            int winR=min_idx/som_dim;
            int winC=(min_idx%som_dim);
            for(int i = 0; i< som_dim*som_dim ; i++){
                int currR=i/som_dim;
                int currC=(i%som_dim);
                float alpha = exp(- sqrt( ((winR-currR)*(winR-currR)) + ((winC-currC)*(winC-currC)) ) / (2*radius*radius));
                for(int j=0; j<no_of_features ; j++){
                    weight_matrix[i* no_of_features+ j]=weight_matrix[i* no_of_features+ j] + ( lrate* alpha *( traindata[rid* no_of_features + j]-weight_matrix[i* no_of_features+ j]));
                }
            }
        }

        if(radius/2.0 <=0)
          radius = 1;
        else
          radius = radius/2.0;

        if(epoch ==epoches-1){
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
        }


    }

    clock_t end = clock();
  	double elapsed_secs_training = double(end - begin) / CLOCKS_PER_SEC;


    cout << endl;
    print_hits(som_dim*som_dim, no_of_classes, hitmap);

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
        printf("class %d \n", class_list[neu]);
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
    cout << "\nClass assignment calculation time in Seconds (find majority voted class and trin accuracy using hitmap) : " << elapsed_secs_calculating_class_assignment <<endl;
    cout << "\nTest accuracy calculation time in Seconds : " << elapsed_secs_testing_accuracy_calc <<endl;


    free(traindata);


    return 0;
}
