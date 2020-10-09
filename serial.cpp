//compile : g++ -o serial.exe serial.cpp -std=c++11
//run : ./Serial.exe

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

  	cout << "SOM data loading";cout << endl;

    clock_t begin = clock();
  	auto started = std::chrono::high_resolution_clock::now();
  	//read and store
  	int no_of_training_data=50;
  	int img_dim=28;
    int no_of_features= img_dim * img_dim;


  	float *traindata= (float*) malloc (no_of_training_data * no_of_features * sizeof(float));
    std::ifstream file("traindata.csv");
    for(int row = 0; row < no_of_training_data; ++row)
    {
          std::string line;
          std::getline(file, line);
          if ( !file.good() )
              break;

          std::stringstream iss(line);

          for (int col = 0; col < no_of_features; ++col)
          {
              std::string val;
              std::getline(iss, val, ',');
              if ( !iss.good() )
                  break;

              std::stringstream convertor(val);
              convertor >> traindata[row * no_of_training_data + col];
          }

    }

  	float *traindata_class= (float*)malloc(no_of_training_data * sizeof(float));
    std::ifstream file2("traindata_class.csv");
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
    }

  	cout << endl;


    // defining SOM
    int som_dim, neu_dim;// SOM network size, neuDim
		float *weight_matrix; //weight matrix
		float lrate;
		float radius;
		int epoch=5;
		int bmu;
		int *hitmap;
		int *class_list; // assigned class for each neuron
		int no_of_classes;
		float train_acc, test_accuracy;

    no_of_classes=10;
		lrate=0.49;
		som_dim = 8; //number of neuron in one dimention
		neu_dim = no_of_features; // input chanels per neuron,kernel size
		radius=som_dim/2.0;
		int no_of_neurons = som_dim*som_dim;
		weight_matrix = (float*)malloc( no_of_neurons * neu_dim * sizeof(float));//number of neurons
		class_list = (int*)malloc(som_dim*som_dim * sizeof(int));
		hitmap = (int*)malloc(som_dim * som_dim * no_of_classes * sizeof(int));//number of neurons

		for(int i = 0; i< som_dim*som_dim; i++){
				for(int j=0; j< no_of_features; j++){
					weight_matrix[i* neu_dim + j] = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.0-0.0)));; // initializing to a random weight
				}
        class_list[i] =0;
		}



    for(int i = 0; i< som_dim*som_dim;i++){
        for(int j=0; j< no_of_classes ; j++){
					hitmap[i  * no_of_classes + j] = 0;// initializing to zero hits
        }
		}

    print_weights(som_dim*som_dim, neu_dim, weight_matrix);
    print_hits(som_dim*som_dim, no_of_classes, hitmap);

    //weight update using the traning trainData

    for (int epoch=0; epoch<1; epoch++)
	  {
		    cout << " epoch" << epoch << endl;
				for(int row = 0; row < 1; ++row)//no_of_training_data
		    {
						//for a  given data record, find the L1 distance to each neuron in SOM
						float *distances = (float*)malloc(no_of_neurons * sizeof(float));
						int min_dis=1000;
						int min_idx=0;
						for(int neu=0; neu<no_of_neurons; neu++){
								distances[neu]=0;
								for(int col=0; col< no_of_features; col++){
										distances[neu] += sqrt( (traindata[row* no_of_features + col] - weight_matrix[ neu * no_of_features + col]) * (traindata[row* no_of_features + col] - weight_matrix[ neu * no_of_features + col])) ;
								}
								printf(" to neu %d distance is %f \n",neu, distances[neu]);

								if(distances[neu]<min_dis){
										min_dis = distances[neu];
										min_idx = neu;
								}
						}

						//find closest neurons
						printf("closest neu %d row %d col %d \n", min_idx, min_idx/som_dim, min_idx%som_dim);

						//update weights
						//int top = ((min_idx/som_dim) -1) !=-1 ? (min_idx/som_dim)-

						//update hitmap and class list after running the last epoch
				}
    }




    return 0;
}
