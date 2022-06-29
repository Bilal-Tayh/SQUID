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
#include <sys/timeb.h>
#include "HeavyHitters_Cuckoo.h"

#define CLK_PER_SEC CLOCKS_PER_SEC


using namespace std;

int main(int argc, char* argv[])
{
    int n = 150000000;
    int k = 1;
    double time = 0;
    
     for (int run = 0; run < k; run++) {
        int* data = (int*) malloc(sizeof(int) * n);
        for (int i = 0; i< n; ++i){
            data[i] = std::rand();
        }
    
        struct timeb begintb, endtb;
        clock_t begint, endt;
        Cuckoo_waterLevel_HH_no_FP_SIMD_256<uint32_t, uint32_t> hh(1, 2, 3, 0.9);
        
        begint = clock();
        ftime(&begintb);
        for (int i = 0; i < n; ++i) {
            hh.insert(i, data[i]);
        }
        endt = clock();
        ftime(&endtb);
        time = ((double)(endt-begint))/CLK_PER_SEC;
     
    
    
    
//      double c =0.0; 
//          
//          
//          
//         for (int i = 0; i< n; ++i){
//             if ( data[i] != hh.query(i)) {
//                 c+=pow(data[i] - hh.query(i),2)/ (data[i]);
//             }
//             
//         }
// 
//         cout<< c << endl ;
     }
        
        
        
    cout  << time/k << endl;

// 	hh.test_correctness();
//  hh.test_speed();

	return 0;
}





