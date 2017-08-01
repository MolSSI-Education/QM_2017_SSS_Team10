#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <omp.h>
#include <math.h>

namespace py = pybind11;


//py::array_t<double> einJ(py::array_t<double> A,
void einJ(py::array_t<double> A,
                         py::array_t<double> B,
			 py::array_t<double> C)
{
    py::buffer_info A_info = A.request();
    py::buffer_info B_info = B.request();
    py::buffer_info C_info = C.request();

    if(A_info.ndim != 4)
	throw std::runtime_error("A is not a 4D tensor");
    if(B_info.ndim != 2)
	throw std::runtime_error("B is not a 2D tensor");

    if(A_info.shape[2] != B_info.shape[0])
	throw std::runtime_error("Dimension mismatch A[2] with B[0]");

    if(A_info.shape[3] != B_info.shape[1])
	throw std::runtime_error("Dimension mismatch A[3] with B[1]");

    size_t C_nrows = A_info.shape[0];
    size_t C_ncols = A_info.shape[1];
    size_t n_k = A_info.shape[2]; 
    size_t n_l = A_info.shape[3]; 
    size_t n_32 = n_l*n_k;
    size_t n_321 = n_32*C_ncols;

    const double* A_data = static_cast<double *>(A_info.ptr);
    const double* B_data = static_cast<double *>(B_info.ptr);
    double* C_data = static_cast<double *>(C_info.ptr);
#pragma omp parallel num_threads(4)
    printf("num threads %d\n", omp_get_thread_num());

//    std::vector<double> C_data(C_nrows*C_ncols);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for(size_t i=0; i< C_nrows; i++){
	for(size_t j=0; j< C_ncols; j++){

//        size_t outer_index = j + i*C_ncols;

	    double val = 0.0;

	    for(size_t k=0; k<n_k; k++){
        for(size_t l=0; l<n_l; l++){
	    val += A_data[i*n_321 + j*n_32 + k*n_l + l] * B_data[l*n_l + k];
        }}
	C_data[i*C_ncols + j] = val;
    }}
}




void einK(py::array_t<double> A,
                         py::array_t<double> B,
			 py::array_t<double> C)
{
    py::buffer_info A_info = A.request();
    py::buffer_info B_info = B.request();
    py::buffer_info C_info = C.request();

    if(A_info.ndim != 4)
	throw std::runtime_error("A is not a 4D tensor");
    if(B_info.ndim != 2)
	throw std::runtime_error("B is not a 2D tensor");

    if(A_info.shape[2] != B_info.shape[0])
	throw std::runtime_error("Dimension mismatch A[2] with B[0]");

    if(A_info.shape[3] != B_info.shape[1])
	throw std::runtime_error("Dimension mismatch A[3] with B[1]");

    size_t C_nrows = A_info.shape[0];
    size_t C_ncols = A_info.shape[1];
    size_t n_k = A_info.shape[2]; 
    size_t n_l = A_info.shape[3]; 
    size_t n_32 = n_l*n_k;
    size_t n_321 = n_32*C_ncols;

    const double* A_data = static_cast<double *>(A_info.ptr);
    const double* B_data = static_cast<double *>(B_info.ptr);
    double* C_data = static_cast<double *>(C_info.ptr);


//    std::vector<double> C_data(C_nrows*C_ncols);
    
#pragma omp parallel for schedule(dynamic)    
    for(size_t i=0; i< C_nrows; i++){
	for(size_t j=0; j< C_ncols; j++){

//        size_t outer_index = j + i*C_ncols;

	    double val = 0.0;

	    for(size_t k=0; k<n_k; k++){
        for(size_t l=0; l<n_l; l++){
	        val += A_data[i*n_321 + k*n_32 + j*n_l + l] * B_data[l*n_l + k];
        }}
        C_data[i*C_ncols + j] = val;
    }}

}



//Define interfaces explicitly
PYBIND11_PLUGIN(basic_mod)
{
    py::module m("basic_mod", "QM10 basic module");
    //Fill module here

    m.def("einJ", &einJ, "Computes Einstein summation necessary to construct Coulomb matrix");
    m.def("einK", &einK, "Computes Einstein summation necessary to construct exchange matrix");

    return m.ptr();
}


