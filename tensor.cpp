#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <fstream>

#include "dais_exc.h"
#include "tensor.h"

#define PI 3.141592654
#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */

using namespace std;


/**
 * Random Initialization
 * 
 * Perform a random initialization of the tensor
 * 
 * @param mean The mean
 * @param std  Standard deviation
 */
void Tensor::init_random(float mean, float std){
    if(data){

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean,std);

        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<d;k++){
                    this->operator()(i,j,k)= distribution(generator);
                }
            }
        }    

    }else{
        throw(tensor_not_initialized());
    }
}

    /**
     * Class constructor
     * 
     * Parameter-less class constructor 
     */
    Tensor::Tensor(){
        throw method_not_implemented();
    }

    /**
     * Class constructor
     * 
     * Creates a new tensor of size r*c*d initialized at value v
     * 
     * @param r
     * @param c
     * @param d
     * @param v
     * @return new Tensor
     */
    Tensor::Tensor(int r, int c, int d, float v){
        init(r,c,d,v);
    }

    /**
     * Class distructor
     * 
     * Cleanup the data when deallocated
     */
    Tensor::~Tensor(){
        throw method_not_implemented();
    }

    /**
     * Operator overloading ()
     * 
     * if indexes are out of bound throw index_out_of_bound() exception
     * 
     * @return the value at location [i][j][k]
     */
    float Tensor::operator()(int i, int j, int k) const{
        //Tensor t;
        //t()
        throw method_not_implemented();
    }

    /**
     * Operator overloading ()
     * 
     * Return the pointer to the location [i][j][k] such that the operator (i,j,k) can be used to 
     * modify tensor data.
     * 
     * If indexes are out of bound throw index_out_of_bound() exception
     * 
     * @return the pointer to the location [i][j][k]
     */
    float &Tensor::operator()(int i, int j, int k){
        throw method_not_implemented();
    }

    /**
     * Copy constructor
     * 
     * This constructor copies the data from another Tensor
     *      
     * @return the new Tensor
     */
    Tensor::Tensor(const Tensor& that){
        throw method_not_implemented();
        //init e poi deep-copy
    }

    /**
     * Operator overloading -
     * 
     * It performs the point-wise difference between two Tensors.
     * 
     * result(i,j,k)=this(i,j,k)-rhs(i,j,k)
     * 
     * The two tensors must have the same size otherwise throw a dimension_mismatch()
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator-(const Tensor &rhs)const{
        throw method_not_implemented();
        //3 for innestati con operazione di sottrazione; PS: Fai overload di ()
    }
    
     /**
     * Operator overloading +
     * 
     * It performs the point-wise sum between two Tensors.
     * 
     * result(i,j,k)=this(i,j,k)+rhs(i,j,k)
     * 
     * The two tensors must have the same size otherwise throw a dimension_mismatch()
     * 
     * @return returns a new Tensor containing the result of the operation
    */
    Tensor Tensor::operator +(const Tensor &rhs)const{
        throw method_not_implemented();
    }

    /**
     * Operator overloading *
     * 
     * It performs the point-wise product between two Tensors.
     * 
     * result(i,j,k)=this(i,j,k)*rhs(i,j,k)
     * 
     * The two tensors must have the same size otherwise throw a dimension_mismatch()
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator*(const Tensor &rhs)const{
        throw method_not_implemented();
    }
    
    /**
     * Operator overloading /
     * 
     * It performs the point-wise division between two Tensors.
     * 
     * result(i,j,k)=this(i,j,k)/rhs(i,j,k)
     * 
     * The two tensors must have the same size otherwise throw a dimension_mismatch()
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator/(const Tensor &rhs)const{
        throw method_not_implemented();
        //attento a divisione per zero, underflow e overflow checks by casting double and then confront F_MAX F_MIN F_NEGATIVE
    }

    /**
     * Operator overloading - 
     * 
     * It performs the point-wise difference between a Tensor and a constant
     * 
     * result(i,j,k)=this(i,j,k)-rhs
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator-(const float &rhs)const{
        throw method_not_implemented();
    }

    /**
     * Operator overloading +
     * 
     * It performs the point-wise sum between a Tensor and a constant
     * 
     * result(i,j,k)=this(i,j,k)+rhs
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator+(const float &rhs)const{
        throw method_not_implemented();
    }

    /**
     * Operator overloading *
     * 
     * It performs the point-wise product between a Tensor and a constant
     * 
     * result(i,j,k)=this(i,j,k)*rhs
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator*(const float &rhs)const{
        throw method_not_implemented();
    }

    /**
     * Operator overloading / between a Tensor and a constant
     * 
     * It performs the point-wise division between a Tensor and a constant
     * 
     * result(i,j,k)=this(i,j,k)/rhs
     * 
     * @return returns a new Tensor containing the result of the operation
     */
    Tensor Tensor::operator/(const float &rhs)const{
        throw method_not_implemented();
    }

    /**
     * Operator overloading = (assignment) 
     * 
     * Perform the assignment between this object and another
     * 
     * @return a reference to the receiver object
     */
    Tensor & Tensor::operator=(const Tensor &other){
        //dealloca e deep-copy
        throw method_not_implemented();
    }


    /**
     * Constant Initialization
     * 
     * Perform the initialization of the tensor to a value v
     * 
     * @param r The number of rows
     * @param c The number of columns
     * @param d The depth
     * @param v The initialization value
     */
    void Tensor::init(int r, int c, int d, float v){
        //3 for innestati, inizializzaione del data con valore "v"
        throw method_not_implemented();
    }

    /**
     * Tensor Clamp
     * 
     * Clamp the tensor such that the lower value becomes low and the higher one become high.
     * 
     * @param low Lower value
     * @param high Higher value 
     */
    void Tensor::clamp(float low, float high){
        //valori tra low e high = uguali così
        //valore < low = low e valore > high = high
        throw method_not_implemented();
    }

    /**
     * Tensor Rescaling
     * 
     * Rescale the value of the tensor following this rule:
     * 
     * newvalue(i,j,k) = ((data(i,j,k)-min(k))/(max(k)-min(k)))*new_max
     * 
     * where max(k) and min(k) are the maximum and minimum value in the k-th channel.
     * 
     * new_max is the new value for the maximum
     * 
     * @param new_max New maximum vale
     */
    void Tensor::rescale(float new_max){
        //trova min e max
        //poi itera e per ciascun value a i,j,k = st
        throw method_not_implemented();
    }

    /**
     * Tensor padding
     * 
     * Zero pad a tensor in height and width, the new tensor will have the following dimensions:
     * 
     * (rows+2*pad_h) x (cols+2*pad_w) x (depth) 
     * 
     * @param pad_h the height padding
     * @param pad_w the width padding
     * @return the padded tensor
     */
    Tensor Tensor::padding(int pad_h, int pad_w)const{
        //con dimensioni del tensore attuale, applico le formule e inizializzo un tensore vuoto con le nuove dimensioni e v a 0
        //ciclo da pad_h/pad_w fino a new_size-pad_h
        throw method_not_implemented();
    }

    /**
     * Subset a tensor
     * 
     * retuns a part of the tensor having the following indices:
     * row_start <= i < row_end  
     * col_start <= j < col_end 
     * depth_start <= k < depth_end
     * 
     * The right extrema is NOT included
     * 
     * @param row_start 
     * @param row_end 
     * @param col_start
     * @param col_end
     * @param depth_start
     * @param depth_end
     * @return the subset of the original tensor
     */
    Tensor Tensor::subset(unsigned int row_start, unsigned int row_end, unsigned int col_start, unsigned int col_end, unsigned int depth_start, unsigned int depth_end)const{
        throw method_not_implemented();
    }

    /** 
     * Concatenate 
     * 
     * The function concatenates two tensors along a give axis
     * 
     * Example: this is of size 10x5x6 and rhs is of 25x5x6
     * 
     * if concat on axis 0 (row) the result will be a new Tensor of size 35x5x6
     * 
     * if concat on axis 1 (columns) the operation will fail because the number 
     * of rows are different (10 and 25).
     * 
     * In order to perform the concatenation is mandatory that all the dimensions 
     * different from the axis should be equal, other wise throw concat_wrong_dimension(). 
     *  
     * @param rhs The tensor to concatenate with
     * @param axis The axis along which perform the concatenation 
     * @return a new Tensor containing the result of the concatenation
     */
    Tensor Tensor::concat(const Tensor &rhs, int axis)const{
        throw method_not_implemented();
    }


    /** 
     * Convolution 
     * 
     * This function performs the convolution of the Tensor with a filter.
     * 
     * The filter f must have odd dimensions and same depth. 
     * 
     * Remeber to apply the padding before running the convolution
     *  
     * @param f The filter
     * @return a new Tensor containing the result of the convolution
     */
    Tensor Tensor::convolve(const Tensor &f)const{
        throw method_not_implemented();
    }

    /* UTILITY */

    /** 
     * Rows 
     * 
     * @return the number of rows in the tensor
     */
    int Tensor::rows()const{
        throw method_not_implemented();
    }

    /** 
     * Cols 
     * 
     * @return the number of columns in the tensor
     */
    int Tensor::cols()const{
        throw method_not_implemented();
    }

    /** 
     * Depth 
     * 
     * @return the depth of the tensor
     */
    int Tensor::depth()const{
        throw method_not_implemented();
    }
    
    /** 
     * Get minimum 
     * 
     * Compute the minimum value considering a particular index in the third dimension
     * 
     * @return the minimum of data( , , k)
     */
    float Tensor::getMin(int k)const{
        //ciclo per i,j e trova il valore minimo del k-esimo canale
        throw method_not_implemented();
    }

    /** 
     * Get maximum 
     * 
     * Compute the maximum value considering a particular index in the third dimension
     * 
     * @return the maximum of data( , , k)
     */
    float Tensor::getMax(int k)const{
        //ciclo per i,j e trova il valore massimo del k-esimo canale
        throw method_not_implemented();
    }

    /** 
     * showSize
     * 
     * shows the dimensions of the tensor on the standard output.
     * 
     * The format is the following:
     * rows" x "colums" x "depth
     * aka cout
     */
    void Tensor::showSize() const{}
    
    /* IOSTREAM */

    /**
     * Operator overloading <<
     * 
     * Use the overaloading of << to show the content of the tensor.
     * 
     * You are free to chose the output format, btw we suggest you to show the tensor by layer.
     * 
     * [..., ..., 0]
     * [..., ..., 1]
     * ...
     * [..., ..., k]
     */
    ostream& operator<< (ostream& stream, const Tensor & obj){
        throw method_not_implemented();
    }

    /**
     * Reading from file
     * 
     * Load the content of a tensor from a textual file.
     * 
     * The file should have this structure: the first three lines provide the dimensions while 
     * the following lines contains the actual data by channel.
     * 
     * For example, a tensor of size 4x3x2 will have the following structure:
     * 4
     * 3
     * 2
     * data(0,0,0)
     * data(0,1,0)
     * data(0,2,0)
     * data(1,0,0)
     * data(1,1,0)
     * .
     * .
     * .
     * data(3,1,1)
     * data(3,2,1)
     * 
     * if the file is not reachable throw unable_to_read_file()
     * 
     * @param filename the filename where the tensor is stored
     */
    void Tensor::read_file(string filename){
        throw method_not_implemented();
    }

    /**
     * Write the tensor to a file
     * 
     * Write the content of a tensor to a textual file.
     * 
     * The file should have this structure: the first three lines provide the dimensions while 
     * the following lines contains the actual data by channel.
     * 
     * For example, a tensor of size 4x3x2 will have the following structure:
     * 4
     * 3
     * 2
     * data(0,0,0)
     * data(0,1,0)
     * data(0,2,0)
     * data(1,0,0)
     * data(1,1,0)
     * .
     * .
     * .
     * data(3,1,1)
     * data(3,2,1)
     * 
     * @param filename the filename where the tensor should be stored
     */
    void Tensor::write_file(string filename){
        throw method_not_implemented();
    }
