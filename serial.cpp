//compile : g++ -o Serial.exe Serial.cpp -std=c++11
//run : ./Serial.exe

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>
#include <iostream>
//#include <string>
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

class SOM {
	public:
		int somDim, neuDim;// SOM network size, neuDim
		float **weights; //weight matrix
		float lrate;
		float radius;
		int epoch=5;
		int bmu;
		int **hitmap;
		int *classList;
		int noOfClasses;int totalTrainCount;float trainAcc;

  };

void print_weights(int rows, int cols, float* array){
  for(int i = 0; i< rows; i++){
      for(int j=0; j< cols ; j++){
        printf("%f " , array[i* cols + j]);
      }
    printf("\n");
  }
}


int main(int argc, char *argv[]){

  	cout << "SOM data loading";cout << endl;
    clock_t begin = clock();
  	auto started = std::chrono::high_resolution_clock::now();
  	//read and store
  	int trainDataSize=500;
  	int img_dim=28;
    int no_of_features= img_dim * img_dim;


  	float traindata[trainDataSize * 784];
      std::ifstream file("traindata.csv");
      for(int row = 0; row < trainDataSize; ++row)
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
              convertor >> traindata[row * trainDataSize + col];
          }

      }

  	  float traindata_class[trainDataSize];
      std::ifstream file2("traindata_class.csv");
      for(int row = 0; row < trainDataSize; ++row)
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
		float *weights; //weight matrix
		float lrate;
		float radius;
		int epoch=5;
		int bmu;
		int *hitmap;
		int *classList;
		int noOfClasses; int totalTrainCount; float trainAcc;

    noOfClasses=10;
		lrate=0.49;
		som_dim = 8; //number of neuron in one dimention
		neu_dim = no_of_features; // input chanels per neuron,kernel size
		radius=somDim/2.0;
		weights = (float*)malloc(som_dim*som_dim * neu_dim * sizeof(float));//number of neurons
		classList = (int*)malloc(som_dim*som_dim * sizeof(int));
		for(int i = 0; i< som_dim*som_dim; i++){
				for(int j=0; j< no_of_features; j++){
					weights[i* neuDim + j] = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.0-0.0)));; // initializing to a random weight
				}
        classList[i] =0 ;
		}

		hitmap = (int*)malloc(som_dim * som_dim * noOfClasses * sizeof(int));//number of neurons

    for(int i = 0; i< som_dim*som_dim;i++){
        for(int j=0; j< noOfClasses ; j++){
					hitmap[i  * noOfClasses + j] = 0;// initializing to zero hits
        }
		}

    print_weights(somDim*somDim, neuDim, weights);

    return 0;
}
