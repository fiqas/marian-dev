// #include<mkl.h>
#include "marian.h"
#include<iostream>
#include <random>
#include <iterator>
#include <cstdint>
#include "3rd_party/cnpy_org/cnpy.h"
#include <iomanip>
#include <algorithm>

float get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}

int main(int argc, char* argv[]) {
  
  std::string model_name = argv[1];
  std::string layer_name = argv[2];
  // int x, inner, y, b, bn;
  int x, inner;
  
  if (layer_name.find("_W1") != std::string::npos) {
    x = 256;
    inner = 1536;
  }
  else if (layer_name.find("_W2") != std::string::npos) {
    x = 1536;
    inner = 256;
  }
  else {
    std::cerr << "Wrong layer name?" << std::endl;
    return -1;
  }

  std::cout << "Shape dense: " << x << " " << inner << std::endl;

  cnpy_org::NpyArray arr_vals = cnpy_org::npz_load(model_name, layer_name);
  auto vals_shape = arr_vals.shape;
  std::cout << "Vals shape: " << vals_shape[0] << " " << vals_shape[1] << std::endl;
  float* A_vals = arr_vals.data<float>();
  std::cout << A_vals[0] << std::endl;
      
  // std::vector<float> A_vals_v(A_vals, A_vals + arr_vals.shape[1]);

  // cnpy_org::NpyArray arr_cols = cnpy_org::npz_load(model_name, "encoder_l1_ffn_W1_cols");
  // int* A_cols = arr_cols.data<int>();
  // std::vector<float> A_cols_v(A_cols, A_cols + arr_cols.shape[1]);

  // cnpy_org::NpyArray arr_ptrB = cnpy_org::npz_load(model_name, "encoder_l1_ffn_W1_ptrB");
  // int* A_ptrB = arr_ptrB.data<int>();
  // std::vector<float> A_ptrB_v(A_ptrB, A_ptrB + arr_ptrB.shape[1]);

  // cnpy_org::NpyArray arr_ptrE = cnpy_org::npz_load(model_name, "encoder_l1_ffn_W1_ptrE");
  // int* A_ptrE = arr_ptrE.data<int>();
  // std::vector<float> A_ptrE_v(A_ptrE, A_ptrE + arr_ptrE.shape[1]);

  int A_rows_num = 256;
  int A_cols_num = 1536;

  std::cout << "Shape: " << A_rows_num << " x " << A_cols_num << std::endl;
  std::cout << "Vector lengths..." << std::endl;
  // std::cout << "Vals: " << A_vals_v.size() << std::endl;
  // std::cout << "Cols: " << A_cols_v.size() << std::endl;
  // std::cout << "PtrB: " << A_ptrB_v.size() << std::endl;
  // std::cout << "PtrE: " << A_ptrE_v.size() << std::endl;


  return 0;
}
