#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <fstream>

#include "dais_exc.h" //just dais_exc.h
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
void Tensor::init_random(float mean, float std)
{
    if (data)
    {

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean, std);

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                for (int k = 0; k < d; k++)
                {
                    this->operator()(i, j, k) = distribution(generator);
                }
            }
        }
    }
    else
    {
        throw(tensor_not_initialized());
    }
}

void destroy(float ***data, int r, int c)
{
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            delete data[i][j];
        }
        delete data[i];
    }
    delete data;
}

/**
 * Class constructor
 * 
 * Parameter-less class constructor 
 */
Tensor::Tensor()
{
    
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
Tensor::Tensor(int r, int c, int d, float v)
{
    init(r, c, d, v);
}

/**
 * Class distructor
 * 
 * Cleanup the data when deallocated
 */
Tensor::~Tensor()
{
    if(data != nullptr) destroy(data, r, c);
}

/**
 * Operator overloading ()
 * 
 * if indexes are out of bound throw index_out_of_bound() exception
 * 
 * @return the value at location [i][j][k]
 */
float Tensor::operator()(int i, int j, int k) const
{
    if(data == nullptr)
        throw tensor_not_initialized();
    if (i >= 0 && i < r && j >= 0 && j < c && k >= 0 && k < d)
    {
        return data[i][j][k];
    }
    else
    {
        throw index_out_of_bound();
    }
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
float &Tensor::operator()(int i, int j, int k)
{
    if(data == nullptr)
        throw tensor_not_initialized();
    if (i >= 0 && i < r && j >= 0 && j < c && k >= 0 && k < d)
    {
        return *(&data[i][j][k]);
    }
    else
    {
        throw index_out_of_bound();
    }
}

/**
 * Copy constructor
 * 
 * This constructor copies the data from another Tensor
 *      
 * @return the new Tensor
 */
Tensor::Tensor(const Tensor &that)
{
    if(that.data != nullptr){
        this->r = that.r;
        this->c = that.c;
        this->d = that.d;
        this->data = new float **[r];
        for (int i = 0; i < r; i++)
        {
            this->data[i] = new float *[c];
            for (int j = 0; j < c; j++)
            {
                this->data[i][j] = new float[d];
                for (int k = 0; k < d; k++)
                {
                    this->data[i][j][k] = that.data[i][j][k];
                }
            }
        }
    } else {
        throw tensor_not_initialized();
    }
}

/**
 * Operator overloading ==
 * 
 * It performs the point-wise equality check between two Tensors.
 * 
 * The equality check between floating points cannot be simply performed using the 
 * operator == but it should take care on their approximation.
 * 
 * This approximation is known as rounding (do you remember "Architettura degli Elaboratori"?)
 *  
 * For example, given a=0.1232 and b=0.1233 they are 
 * - the same, if we consider a rounding with 1, 2 and 3 decimals 
 * - different when considering 4 decimal points. In this case b>a
 * 
 * So, given two floating point numbers "a" and "b", how can we check their equivalence? 
 * through this formula:
 * 
 * a ?= b if and only if |a-b|<EPSILON
 * 
 * where EPSILON is fixed constant (defined at the beginning of this header file)
 * 
 * Two tensors A and B are the same if:
 * A[i][j][k] == B[i][j][k] for all i,j,k 
 * where == is the above formula.
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns true if all their entries are "floating" equal
 */
bool Tensor::operator==(const Tensor &rhs) const
{
    if (rhs.data == nullptr){
        if(data == nullptr){
            return true;
        }
        return false;
    }
    if (this->r != rhs.r || this->c != rhs.c || this->d != rhs.d)
    {
        throw dimension_mismatch();
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (!(abs(this->data[i][j][k] - rhs(i, j, k))<EPSILON))
                    return false;
            }
        }
    }
    return true;
}

/**
 * Check if there is overflow or underflow
 * 
 * @return returns true if there is overflow or underflow false if otherwise
 * 
 */
bool operationFlowCheck(double result)
{
    //first half checks positive overflow/underflow and second part checks the negative ones
    return (result >= FLT_MAX || (result <= FLT_MIN && result > 0)) || (result <= -FLT_MAX || (result >= -FLT_MIN && result < 0));
}

/**
 * It performs the point-wise difference between two Tensors.
 * 
 * result(i,j,k)=this(i,j,k)-rhs(i,j,k)
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator-(const Tensor &rhs) const
{
    if(data == nullptr || rhs.data == nullptr)
        throw tensor_not_initialized();
    if (this->r != rhs.r || this->c != rhs.c || this->d != rhs.d)
    {
        throw dimension_mismatch();
    }
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] - (double)rhs(i, j, k)))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] - rhs(i, j, k);
            }
        }
    }
    return result;
}

/**
 * 
 * It performs the point-wise sum between two Tensors.
 * 
 * result(i,j,k)=this(i,j,k)+rhs(i,j,k)
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns a new Tensor containing the result of the operation
*/
Tensor Tensor::operator+(const Tensor &rhs) const
{
    if(data == nullptr || rhs.data == nullptr)
        throw tensor_not_initialized();
    if (this->r != rhs.r || this->c != rhs.c || this->d != rhs.d)
    {
        throw dimension_mismatch();
    }
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] + (double)rhs(i, j, k)))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] + rhs(i, j, k);
            }
        }
    }
    return result;
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
Tensor Tensor::operator*(const Tensor &rhs) const
{
    if(data == nullptr || rhs.data == nullptr)
        throw tensor_not_initialized();
    if (this->r != rhs.r || this->c != rhs.c || this->d != rhs.d)
    {
        throw dimension_mismatch();
    }
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] * (double)rhs(i, j, k)))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] * rhs(i, j, k);
            }
        }
    }
    return result;
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
Tensor Tensor::operator/(const Tensor &rhs) const
{
    if(data == nullptr || rhs.data == nullptr)
        throw tensor_not_initialized();
    if (this->r != rhs.r || this->c != rhs.c || this->d != rhs.d)
    {
        throw dimension_mismatch();
    }
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (rhs(i, j, k) == 0.0)
                    throw division_by_zero();
                else if (operationFlowCheck((double)this->data[i][j][k] / (double)rhs(i, j, k)))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] / rhs(i, j, k);
            }
        }
    }
    return result;
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
Tensor Tensor::operator-(const float &rhs) const
{
    if(data == nullptr) 
        throw tensor_not_initialized();
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] - (double)rhs))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] - rhs;
            }
        }
    }

    return result;
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
Tensor Tensor::operator+(const float &rhs) const
{
    if(data == nullptr) 
        throw tensor_not_initialized();
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] - (double)rhs))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] + rhs;
            }
        }
    }

    return result;
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
Tensor Tensor::operator*(const float &rhs) const
{
    if(data == nullptr) 
        throw tensor_not_initialized();
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] - (double)rhs))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] * rhs;
            }
        }
    }

    return result;
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
Tensor Tensor::operator/(const float &rhs) const
{
    if(data == nullptr) 
        throw tensor_not_initialized();
    if (rhs == 0.0)
        throw division_by_zero();
    Tensor result(this->r, this->c, this->d, 0.0);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (operationFlowCheck((double)this->data[i][j][k] - (double)rhs))
                    throw error_in_operation();
                else
                    result(i, j, k) = this->data[i][j][k] / rhs;
            }
        }
    }

    return result;
}

/**
 * Operator overloading = (assignment) 
 * 
 * Perform the assignment between this object and another
 * 
 * @return a reference to the receiver object
 */
Tensor &Tensor::operator=(const Tensor &other)
{
    if(other.data == nullptr)
        throw tensor_not_initialized();
    destroy(data, r, c);
    this->r = other.r;
    this->c = other.c;
    this->d = other.d;
    this->data = new float **[r];
    for (int i = 0; i < r; i++)
    {
        this->data[i] = new float *[c];
        for (int j = 0; j < c; j++)
        {
            this->data[i][j] = new float[d];
            for (int k = 0; k < d; k++)
            {
                this->data[i][j][k] = other(i,j,k);
            }
        }
    }
    return *this;
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
void Tensor::init(int r, int c, int d, float v)
{
    //3 for innestati, inizializzaione del data con valore "v"
    if(r == 0 || c == 0 || d == 0) 
        throw tensor_not_initialized();
    this->r = r;
    this->c = c;
    this->d = d;
    data = new float **[r];
    for (int i = 0; i < r; i++)
    {
        data[i] = new float *[c];
        for (int j = 0; j < c; j++)
        {
            data[i][j] = new float[d];
            for (int k = 0; k < d; k++)
            {
                data[i][j][k] = v;
            }
        }
    }
}

/**
 * Tensor Clamp
 * 
 * Clamp the tensor such that the lower value becomes low and the higher one become high.
 * 
 * @param low Lower value
 * @param high Higher value 
 */
void Tensor::clamp(float low, float high)
{
    if(data == nullptr) throw tensor_not_initialized();
    //valori tra low e high = uguali così
    //valore < low = low e valore > high = high
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < d; k++)
            {
                if (data[i][j][k] < low)
                    data[i][j][k] = low;
                else if (data[i][j][k] > high)
                    data[i][j][k] = high;
            }
        }
    }
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
 * new_max is the new maximum value for each channel
 * 
 * - if max(k) and min(k) are the same, then the entire k-th channel is set to new_max.
 * 
 * @param new_max New maximum vale
 */
void Tensor::rescale(float new_max)
{
    if(data == nullptr)
        throw tensor_not_initialized();
    for (int k = 0; k < d; k++)
    {
        float min = getMin(k);
        float max = getMax(k);
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                if(min == max)
                    this->data[i][j][k] = new_max;
                else
                    this->data[i][j][k] = (this->data[i][j][k] - min) / (max - min) * new_max;
            }
        }
    }
    //TODO debugging
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
Tensor Tensor::padding(int pad_h, int pad_w) const
{
    if(data == nullptr)
        throw tensor_not_initialized();
    int rows = this->r + 2 * pad_h;
    int cols = this->c + 2 * pad_w;
    int depth = this->d;
    Tensor padded(rows, cols, depth, 0.0);
    for (int i = pad_h; i < rows - pad_h; i++)
    {
        for (int j = pad_w; j < cols - pad_w; j++)
        {
            for (int k = 0; k < depth; k++)
            {
                float dataToPass = this->data[i - pad_w][j - pad_h][k];
                padded(i, j, k) = dataToPass;
            }
        }
    }
    return padded;
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
Tensor Tensor::subset(unsigned int row_start, unsigned int row_end, unsigned int col_start, unsigned int col_end, unsigned int depth_start, unsigned int depth_end) const
{
    if(data == nullptr)
        throw tensor_not_initialized();
    unsigned int depth = this->d, rows = this->r, cols = this->c;
    if (!((depth >= depth_start && depth >= depth_end) &&
          (cols >= col_start && cols >=  col_end) &&
          (rows >= row_start && rows >=  row_end)) &&
        (depth_end >= depth_start && row_end >= row_start && col_end >= col_start))
        throw index_out_of_bound();

    Tensor result(row_end - row_start, col_end - col_start, depth_end - depth_start, 0.0);

    for (unsigned int i = row_start; i < row_end; i++)
    {
        for (unsigned int j = col_start; j < col_end; j++)
        {
            for (unsigned int k = depth_start; k < depth_end; k++)
            {
                result(i - row_start, j - col_start, k - depth_start) = this->data[i][j][k];
            }
        }
    }
    return result;
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
Tensor Tensor::concat(const Tensor &rhs, int axis) const
{
    if(rhs.data==nullptr || data==nullptr)throw tensor_not_initialized();
    if(axis < 0 || axis > 2) throw concat_wrong_dimension();
    if(axis==0 && rhs.c!=this->c && rhs.d!=this->d)throw dimension_mismatch();
    if(axis==1 && rhs.r!=this->r && rhs.d!=this->d)throw dimension_mismatch();
    if(axis==2 && rhs.r!=this->r && rhs.c!=this->c)throw dimension_mismatch();
    int rows, cols, depth;
    switch(axis){
        case 0:
            rows = this->r + rhs.r;
            cols = this->c;
            depth = this->d;
            break;
        case 1:
            rows = this->r;
            cols = this->c + rhs.c;
            depth = this->d;
            break;
        case 2:
            rows = this->r;
            cols = this->c;
            depth = this->d + rhs.d;
            break;
    }
    Tensor result(rows,cols,depth,0.0);
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            for(int k = 0; k<depth; k++){
                switch(axis){
                    case 0:
                        if(i<this->r){
                            result(i,j,k) = this->data[i][j][k];
                        } else {
                            result(i,j,k) = rhs(i-this->r,j,k);
                        }
                        break;
                    case 1:
                        if(j<this->c){
                            result(i,j,k) = this->data[i][j][k];
                        } else {
                            result(i,j,k) = rhs(i,j-this->c,k);
                        }
                        break;
                    case 2:
                        if(k<this->d){
                            result(i,j,k) = this->data[i][j][k];
                        } else {
                            result(i,j,k) = rhs(i,j,k-this->d);
                        }
                        break;
                }
            }
        }
    }
    return result;
}

float applyFilter(const Tensor &a,const Tensor &b, int filterChannel){
    float result=0.0;
    for(int i = 0; i<b.rows(); i++){
        for(int j = 0; j<a.cols(); j++){
            result+=a(i,j,0)*b(i,j,filterChannel);
        }
    }
    return result;
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
Tensor Tensor::convolve(const Tensor &f) const
{
    if(f.data==nullptr || data==nullptr) throw tensor_not_initialized();
    if(this->d!=f.d)throw dimension_mismatch();
    Tensor padded;
    Tensor result(*this);
    padded = result.padding(floor(f.rows()/2),floor(f.cols()/2));
    for(int i = 0; i<result.rows(); i++){
        for(int j = 0; j<result.cols(); j++){
            for(int k = 0; k<result.depth(); k++){
                Tensor toApplyFilterOn = padded.subset(i, i+f.rows(), j, j+f.cols(), k, k+1);
                result(i,j,k) = applyFilter(toApplyFilterOn, f, k);
            }
        }
    }
    return result;
}

/* UTILITY */

/** 
 * Rows 
 * 
 * @return the number of rows in the tensor
 */
int Tensor::rows() const
{
    return r;
}

/** 
 * Cols 
 * 
 * @return the number of columns in the tensor
 */
int Tensor::cols() const
{
    return c;
}

/** 
 * Depth 
 * 
 * @return the depth of the tensor
 */
int Tensor::depth() const
{
    return d;
}

/** 
 * Get minimum 
 * 
 * Compute the minimum value considering a particular index in the third dimension
 * 
 * @return the minimum of data( , , k)
 */
float Tensor::getMin(int k) const
{
    if(data == nullptr) throw tensor_not_initialized();
    if(k < 0 || k > depth()) throw index_out_of_bound();
    float min = FLT_MAX;
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            if (min > data[i][j][k])
            {
                min = data[i][j][k];
            }
        }
    }
    return min;
}

/** 
 * Get maximum 
 * 
 * Compute the maximum value considering a particular index in the third dimension
 * 
 * @return the maximum of data( , , k)
 */
float Tensor::getMax(int k) const
{
    if(data == nullptr) throw tensor_not_initialized();
    if(k < 0 || k > depth()) throw index_out_of_bound();
    float max = FLT_MIN;
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            if (max < data[i][j][k])
            {
                max = data[i][j][k];
            }
        }
    }
    return max;
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
void Tensor::showSize() const
{
    std::cout << r << " x " << c << " x " << d << std::endl;
}

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
ostream &operator<<(ostream &stream, const Tensor &obj)
{
    if(obj.data==nullptr) throw tensor_not_initialized();
    for(int k = 0; k<obj.depth(); k++){
        stream << "Layer " << k << " " <<endl;
        for(int i = 0; i<obj.rows(); i++){
            for(int j = 0; j<obj.cols(); j++){
                stream << "[" << i << ", " << j << " ] =>" << obj(i,j,k) << endl;
            }
        }
        stream << "End of Layer " << k << endl;
    }
    return stream;
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
void Tensor::read_file(string filename)
{
    if(data != nullptr) destroy(data, r, c);
    ifstream file(filename);
    if(file.good()){
        string result;
        file >> r;
        file >> c;
        file >> d;
        init(r,c,d,0.0);
        for (int k = 0; k < d && !file.eof(); k++)
        {
            for (int i = 0; i < r && !file.eof(); i++)
            {
                for (int j = 0; j < c && !file.eof(); j++)
                {
                    file >> this->data[i][j][k];
                }
            }
        }
        file.close();
    }else throw unable_to_read_file();
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
void Tensor::write_file(string filename)
{
    if(data == nullptr) throw tensor_not_initialized();
    ofstream output{filename};
    if(!output.good())throw unable_to_open_file();
        output<<this->r<<endl;
        output<<this->c<<endl;
        output<<this->d<<endl;
        for (int k = 0; k < this->d ; k++)
        {
            for (int i = 0; i < this->r ; i++)
            {
                for (int j = 0; j < this->c ; j++)
                {
                     output<<this->data[i][j][k]<<endl;
                }
            }
        }
        output.close();
}