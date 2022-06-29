#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <math.h>
#include <fstream>
#include <assert.h>
#include <time.h>
#include <sstream>
#include <string>
#include <cstring>
#include <unordered_map>
// #include <windows.h>
// #include "Cuckoo.h"
//#include "Cuckoo_water_level.h"
#include <stdint.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <sys/timeb.h>
#include "Utils.hpp"
#include "HeavyHitters_Cuckoo.h"

#define CLK_PER_SEC CLOCKS_PER_SEC
#define CAIDA16_SIZE 152197439
#define CAIDA18_SIZE 175880896
#define UNIV1_SIZE 17323447

typedef unsigned long long key;
typedef double val;

using namespace std;


// void getKeysFromFile(string filename, vector<key*> &keys, int size) {
//   ifstream stream;
//   stream.open(filename, fstream::in | fstream::out | fstream::app);
//   if (!stream) {
//     throw invalid_argument("Could not open " + filename + " for reading.");
//   }
// 
//   key* data = (key*) malloc(sizeof(key) * size);
//   string line;
//   for (int i = 0; i< size; ++i){
//     getline(stream, line);
//     try {
//       data[i] = stoull(line);
//     } catch (const invalid_argument& ia) {
//       cerr << "Invalid argument: " << ia.what() << " at line " << i << endl;
//       cerr << line << endl;
//       --i;
//     }
//   }
// 
//   keys.push_back(data);
// 
//   stream.close();
// }




void getKeysAndWeightsFromFile(string filename, vector<key*> &keys, vector<val*> &value, int size) {
  ifstream stream;
  stream.open(filename, fstream::in | fstream::out | fstream::app);
  if (!stream) {
    throw invalid_argument("Could not open " + filename + " for reading.");
  }

  key* file_keys = (key*) malloc(sizeof(key) * size);
  val* file_ws = (val*) malloc(sizeof(val) * size);

  string line;
  string len;
  string id;
  for (int i = 0; i < size; ++i){
    getline(stream, line);
    std::istringstream iss(line);
    iss >> len;
    iss >> id;
    try {
      file_keys[i] = stoull(id);
      file_ws[i] = stod(len);
    } catch (const std::invalid_argument& ia) {
      cerr << "Invalid argument: " << ia.what() << " at line " << i << endl;
      cerr << len << " " << id << endl;;
      --i;
      exit(1);
    }
  }
  keys.push_back(file_keys);
  value.push_back(file_ws);

  stream.close();
}




int main(int argc, char* argv[])
{
    vector<key*> keys;
    vector<val*> values;
    vector<int> sizes;
    vector<string> datasets;
    int Dset = 2;
    
    
    if(Dset == 1){
        getKeysAndWeightsFromFile("../../datasets/UNIV1/mergedAggregatedPktlen_Srcip", keys, values, UNIV1_SIZE);
        sizes.push_back(UNIV1_SIZE);
        datasets.push_back("univ1");
    }

    if(Dset == 2){
        getKeysAndWeightsFromFile("../../datasets/CAIDA16/mergedAggregatedPktlen_Srcip", keys, values, CAIDA16_SIZE);
        sizes.push_back(CAIDA16_SIZE);
        datasets.push_back("caida");
    }

    if(Dset == 3){
        getKeysAndWeightsFromFile("../../datasets/CAIDA18/mergedAggregatedPktlen_Srcip", keys, values, CAIDA18_SIZE);
        sizes.push_back(CAIDA18_SIZE);
        datasets.push_back("caida18");
    }
    
    
    
    int k = 1;
    double time = 0;
    double nrmse = 0;
    double maxl = 0.7;
    double gamma = 0.99;

    
    for (int run = 0; run < k; run++) {
        vector<key*>::iterator k_it = keys.begin();
        vector<val*>::iterator v_it = values.begin();
        vector<int>::iterator s_it = sizes.begin();
        vector<string>::iterator d_it = datasets.begin();
        
        for (int trc = 0; trc < 1; trc++) {
            key* kk = *k_it;
            val* vv = *v_it;
            int size = *s_it;
            string dataset = *d_it;
            
//             cout<< "****  "<<dataset<<"  ****" << endl ;
            
            
            struct timeb begintb, endtb;
            clock_t begint, endt;
            Cuckoo_waterLevel_HH_no_FP_SIMD_256<uint32_t, uint32_t> hh(1, 2, 3, maxl);
            begint = clock();
            ftime(&begintb);
            for (int i = 0; i < size; ++i) {
                hh.insert(kk[i], vv[i]);
//                 cout<<i << endl;
            }
//             cout<<"outtttttttt1" << endl;
            endt = clock();
            ftime(&endtb);
            time += ((double)(endt-begint))/CLK_PER_SEC;
            ++k_it;
            ++v_it;
            ++s_it;
            ++d_it;
            
            
                
//             double c =0.0; 
//             uint64_t vol =0;
//             unordered_map<key, val> map;
//             Cuckoo_waterLevel_HH_no_FP_SIMD_256<uint32_t, uint32_t> hh1(1, 2, 3, maxl,gamma);
//             for (int i = 0; i < size; ++i) {
//                     map[kk[i]] +=vv[i];
//                     hh1.insert(kk[i], vv[i]);
//                     uint32_t err = map[kk[1]] - hh1.query(kk[i]);
//                     c+= err*err;
//                     vol+=vv[i];
//             }
//             double mse = c/size;
//             double rmse = sqrt(mse);
//             nrmse += rmse/vol;
        }
        
    }
    cout  << time/k << endl;
    cout  << nrmse/k << endl;
    return 0;   
    
}





