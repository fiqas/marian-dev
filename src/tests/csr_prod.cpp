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
  // int A_rows_num, A_cols_num, y, b, bn;
  int A_rows_num, A_cols_num;
  
  if (layer_name.find("_W1") != std::string::npos) {
    A_rows_num = 256;
    A_cols_num = 1536;
  }
  else if (layer_name.find("_W2") != std::string::npos) {
    A_rows_num = 1536;
    A_cols_num = 256;
  }
  else {
    std::cerr << "Wrong layer name?" << std::endl;
    return -1;
  }

  std::cout << "Shape dense: " << A_rows_num << " " << A_cols_num << std::endl;
  
  ///////////////////
  cnpy_org::NpyArray arr_dense = cnpy_org::npz_load(model_name, layer_name);
  auto dense_shape = arr_dense.shape;  
  std::cout << "Dense shape: " << dense_shape[0] << " " << dense_shape[1] << std::endl;
  
  float* A_dense = arr_dense.data<float>();
  std::cout << A_dense[0] << std::endl;
      
  std::cout << "A_dense" << std::endl;
  for (size_t i = 0; i < 100; i++) {
    std::cout << A_dense[i] << " ";
  }
  std::cout << std::endl;
  std::vector<float> A_dense_v(A_dense, A_dense + arr_dense.shape[0] * arr_dense.shape[1]);
  ///////////////////

  ///////////////////
  cnpy_org::NpyArray arr_vals = cnpy_org::npz_load(model_name, layer_name + "_vals");
  auto vals_shape = arr_vals.shape;  
  std::cout << "Vals shape: " << vals_shape[0] << " " << vals_shape[1] << std::endl;
  
  float* A_vals = arr_vals.data<float>();
  std::cout << A_vals[0] << std::endl;
      
  std::cout << "A_vals" << std::endl;
  for (size_t i = 0; i < 100; i++) {
    std::cout << A_vals[i] << " ";
  }
  std::cout << std::endl;
  std::vector<float> A_vals_v(A_vals, A_vals + arr_vals.shape[1]);
  ///////////////////
  
  ///////////////////
  cnpy_org::NpyArray arr_cols = cnpy_org::npz_load(model_name, layer_name + "_cols");
  auto cols_shape = arr_cols.shape;  
  std::cout << "Vals shape: " << cols_shape[0] << " " << cols_shape[1] << std::endl;
  
  int* A_cols = arr_cols.data<int>();
  std::cout << A_cols[0] << std::endl;
      
  std::cout << "A_cols" << std::endl;
  for (size_t i = 0; i < 100; i++) {
    std::cout << A_cols[i] << " ";
  }
  std::cout << std::endl;
  std::vector<int> A_cols_v(A_cols, A_cols + arr_cols.shape[1]);
  ///////////////////

  ///////////////////
  cnpy_org::NpyArray arr_ptrB = cnpy_org::npz_load(model_name, layer_name + "_ptrB");
  auto ptrB_shape = arr_ptrB.shape;  
  std::cout << "Vals shape: " << ptrB_shape[0] << " " << ptrB_shape[1] << std::endl;
  
  int* A_ptrB = arr_ptrB.data<int>();
  std::cout << A_ptrB[0] << std::endl;
      
  std::cout << "A_ptrB" << std::endl;
  for (size_t i = 0; i < 100; i++) {
    std::cout << A_ptrB[i] << " ";
  }
  std::cout << std::endl;
  std::vector<int> A_ptrB_v(A_ptrB, A_ptrB + arr_ptrB.shape[1]);
  ///////////////////
  
  ///////////////////
  cnpy_org::NpyArray arr_ptrE = cnpy_org::npz_load(model_name, layer_name + "_ptrE");
  auto ptrE_shape = arr_ptrE.shape;  
  std::cout << "Vals shape: " << ptrE_shape[0] << " " << ptrE_shape[1] << std::endl;
  
  int* A_ptrE = arr_ptrE.data<int>();
  std::cout << A_ptrE[0] << std::endl;
      
  std::cout << "A_ptrE" << std::endl;
  for (size_t i = 0; i < 100; i++) {
    std::cout << A_ptrE[i] << " ";
  }
  std::cout << std::endl;
  std::vector<int> A_ptrE_v(A_ptrE, A_ptrE + arr_ptrE.shape[1]);
  ///////////////////
  
  std::cout << "Shape: " << A_rows_num << " A_rows_num " << A_cols_num << std::endl;
  std::cout << "Vector lengths..." << std::endl;
  std::cout << "Vals: " << A_vals_v.size() << std::endl;
  std::cout << "Cols: " << A_cols_v.size() << std::endl;
  std::cout << "PtrB: " << A_ptrB_v.size() << std::endl;
  std::cout << "PtrE: " << A_ptrE_v.size() << std::endl;



  //////////////////////////////////////////////////////////
  //
  
  using namespace marian; {
   
   
  auto g = New<ExpressionGraph>(true);
  g->setDevice({0, DeviceType::cpu});
  g->getBackend()->setOptimized(false);
  g->reserveWorkspaceMB(2512);

  g->clear();

  auto x = g->constant({1536, 512}, inits::glorotUniform());
  
  std::cerr << "x shape " << x->shape() << std::endl;
  std::cerr << "W1 shape " << A_rows_num << " " << A_cols_num << std::endl;

  
  auto W1_dense = g->param("W1_dense", {(int)dense_shape[0], (int)dense_shape[1]}, inits::fromVector(A_dense_v), Type::float32);

  auto W1_vals = g->param("W1_vals", {(int)A_vals_v.size()}, inits::fromVector(A_vals_v), Type::float32);
  auto W1_cols = g->param("W1_cols", {(int)A_cols_v.size()}, inits::fromVector(A_cols_v), Type::int32);
  auto W1_ptrB = g->param("W1_ptrB", {(int)A_ptrB_v.size()}, inits::fromVector(A_ptrB_v), Type::int32);
  auto W1_ptrE = g->param("W1_ptrE", {(int)A_ptrE_v.size()}, inits::fromVector(A_ptrE_v), Type::int32);

  auto output_dense = dot(W1_dense, x);
  debug(output_dense);  

  auto output = mkl_csr_dot( // sparse x dense
	                    {A_rows_num, A_cols_num}, // S shape
			    W1_vals,
			    W1_cols,
			    W1_ptrB,
			    W1_ptrE,
			    x,
			    false);

  debug(output);
  std::cerr << "output shape: " << output->shape() << std::endl;
  g->forward();
  
  auto b1 = g->param("b1", {1, 1536}, inits::glorotUniform());
 

  ////////////////////////////////////////
  //
  }

  return 0;
}
