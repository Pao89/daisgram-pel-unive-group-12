#include <iostream>
#include <string>

#include "dais_exc.h"
#include "tensor.h"
#include "libbmp.h"
#include "DAISGram.h"

using namespace std;

/**
 * Load a bitmap from file
 *
 * @param filename String containing the path of the file
 */
void DAISGram::load_image(string filename){
    BmpImg img = BmpImg();

    img.read(filename.c_str());

    const int h = img.get_height();
    const int w = img.get_width();

    data = Tensor(h, w, 3, 0.0);

    for(int i=0;i<img.get_height();i++){
        for(int j=0;j<img.get_width();j++){ 
            data(i,j,0) = (float) img.red_at(j,i);
            data(i,j,1) = (float) img.green_at(j,i);    
            data(i,j,2) = (float) img.blue_at(j,i);   
        }                
    }
}


/**
 * Save a DAISGram object to a bitmap file.
 * 
 * Data is clamped to 0,255 before saving it.
 *
 * @param filename String containing the path where to store the image.
 */
void DAISGram::save_image(string filename){

    data.clamp(0,255);

    BmpImg img = BmpImg(getCols(), getRows());

    img.init(getCols(), getRows());

    for(int i=0;i<getRows();i++){
        for(int j=0;j<getCols();j++){
            img.set_pixel(j,i,(unsigned char) data(i,j,0),(unsigned char) data(i,j,1),(unsigned char) data(i,j,2));                   
        }                
    }

    img.write(filename);

}


/**
 * Generate Random Image
 * 
 * Generate a random image from nois
 * 
 * @param h height of the image
 * @param w width of the image
 * @param d number of channels
 * @return returns a new DAISGram containing the generated image.
 */  
void DAISGram::generate_random(int h, int w, int d){
    data = Tensor(h,w,d,0.0);
    data.init_random(128,50);
    data.rescale(255);
}

DAISGram::DAISGram(){
    
}

DAISGram::~DAISGram(){
    
}

/**
 * Get rows
 *
 * @return returns the number of rows in the image
 */
int DAISGram::getRows(){
    return data.rows();
}

/**
 * Get columns
 *
 * @return returns the number of columns in the image
 */
int DAISGram::getCols(){
    return data.cols();
}

/**
 * Get depth
 *
 * @return returns the number of channels in the image
 */
int DAISGram::getDepth(){
    return data.depth();
}

/**
 * Brighten the image
 * 
 * It sums the bright variable to all the values in the image.
 * 
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 * 
 * @param bright the amount of bright to add (if negative the image gets darker)
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::brighten(float bright){
    DAISGram result;
    Tensor resultData(data);
    resultData = resultData + bright;
    resultData.clamp(0,255);
    result.data = resultData;
    return result;
}

/**
 * Create a grayscale version of the object
 * 
 * A grayscale image is produced by substituting each pixel with its average on all the channel
 *  
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::grayscale(){
    DAISGram result;
    Tensor resultData(data);
    for(int i=0;i<getRows();i++){
        for(int j=0;j<getCols();j++){
            float avg = 0.0;
            for(int k=0;k<getDepth();k++){
                avg += resultData(i,j,k);         
            }
            avg = avg/getDepth();
            for(int k=0;k<getDepth();k++){
                resultData(i,j,k) = avg;     
            }         
        }                
    }
    resultData.clamp(0,255);
    result.data = resultData;
    return result;
}


void swap(float& n1, float& n2){
    float temp = n1;
    n1 = n2;
    n2 = temp;
}

/**
 * switchChannels
 * 
 * swaps the channels (usually depth) of a given Tensor
*/
Tensor switchChannels(Tensor data, int c1, int c2){
    Tensor result = data;
    for(int i = 0; i<result.rows(); i++){
        for(int j = 0; j<result.cols(); j++){
            swap(result(i,j,c1), result(i,j,c2));
        }
    }
    return result;
}

/**
 * Create a Warhol effect on the image
 * 
 * This function returns a composition of 4 different images in which the:
 * - top left is the original image
 * - top right is the original image in which the Red and Green channel are swapped
 * - bottom left is the original image in which the Blue and Green channel are swapped
 * - bottom right is the original image in which the Red and Blue channel are swapped
 *  
 * The output image is twice the dimensions of the original one.
 * 
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::warhol(){
    DAISGram result;
    Tensor topLeft = data;
    Tensor topRight = switchChannels(data, 0, 1);
    Tensor bottomLeft = switchChannels(data, 1, 2);
    Tensor bottomRight = switchChannels(data, 0, 2);
    Tensor resultData;
    topLeft = topLeft.concat(topRight, 1);
    bottomLeft = bottomLeft.concat(bottomRight, 1);
    resultData = topLeft.concat(bottomLeft, 0);
    result.data = resultData;
    return result;
}



/**
 * Sharpen the image
 * 
 * This function makes the image sharper by convolving it with a sharp filter
 * 
 * filter[3][3]
 *    0  -1  0
 *    -1  5 -1
 *    0  -1  0
 *  
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 * 
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::sharpen(){
    DAISGram result;
    Tensor resultData(data);
    Tensor filter(3,3,resultData.depth(), -1.0);
    for(int k = 0; k<filter.depth(); k++){
        filter(0,0,k) = 0;
        filter(0,2,k) = 0;
        filter(1,1,k) = 5;
        filter(2,0,k) = 0;
        filter(2,2,k) = 0;
    }
    resultData = resultData.convolve(filter);
    resultData.clamp(0,255);
    result.data = resultData;
    return result;
}

/**
 * Emboss the image
 * 
 * This function makes the image embossed (a light 3D effect) by convolving it with an
 * embossing filter
 * 
 * filter[3][3]
 *    -2 -1  0
 *    -1  1  1
 *     0  1  2
 * 
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *  
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::emboss(){
    DAISGram result;
    Tensor resultData(data);
    Tensor filter(3,3,resultData.depth(), 1.0);
    for(int k = 0; k<filter.depth(); k++){
        filter(0,0,k) = -2;
        filter(0,1,k) = -1;
        filter(0,2,k) = 0;
        filter(1,0,k) = -1;
        filter(2,0,k) = 0;
        filter(2,2,k) = 2;
    }
    resultData = resultData.convolve(filter);
    resultData.clamp(0,255);
    result.data = resultData;
    return result;
}

/**
 * Smooth the image
 * 
 * This function remove the noise in an image using convolution and an average filter
 * of size h*h:
 * 
 * c = 1/(h*h)
 * 
 * filter[3][3]
 *    c c c
 *    c c c
 *    c c c
 *  
 * @param h the size of the filter
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::smooth(int h){
    float smoothener = h;
    Tensor filter(h,h,data.depth(),(1.0/(smoothener*smoothener)));
    DAISGram result;
    Tensor resultData(data);
    resultData = resultData.convolve(filter);
    result.data = resultData;
    return result;
}

/**
 * Edges of an image
 * 
 * This function extract the edges of an image by using the convolution 
 * operator and the following filter
 * 
 * 
 * filter[3][3]
 * -1  -1  -1
 * -1   8  -1
 * -1  -1  -1
 * 
 * Remeber to convert the image to grayscale before running the convolution.
 * 
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *  
 * @return returns a new DAISGram containing the modified object
 */  
DAISGram DAISGram::edge(){
    DAISGram result;
    result = grayscale();
    Tensor edgened(result.data);
    Tensor filter(3,3,edgened.depth(),-1.0);
    for(int i = 0; i<edgened.depth(); i++){
        filter(1,1,i) = 8.0;
    }
    edgened = edgened.convolve(filter);
    edgened.clamp(0,255);
    result.data = edgened;
    return result;
}

/**
 * Blend with anoter image
 * 
 * This function generate a new DAISGram which is the composition 
 * of the object and another DAISGram object
 * 
 * The composition follows this convex combination:
 * results = alpha*this + (1-alpha)*rhs 
 * 
 * rhs and this obejct MUST have the same dimensions.
 * 
 * @param rhs The second image involved in the blending
 * @param alpha The parameter of the convex combination  
 * @return returns a new DAISGram containing the blending of the two images.
 */  
DAISGram DAISGram::blend(const DAISGram & rhs, float alpha){
    if (getRows() != rhs.data.rows() || getCols() != rhs.data.cols() || getDepth() != rhs.data.depth())
    {
        throw dimension_mismatch();
    }
    DAISGram result;
    Tensor resultData(data);
    for(int i = 0; i<getRows(); i++){
        for(int j = 0; j<getCols(); j++){
            for(int k = 0; k<getDepth(); k++){
                resultData(i,j,k) = (alpha*resultData(i,j,k)) + ((1.0-alpha)*rhs.data(i,j,k)) ;//alpha*resultData(i,j,k) + (1.0-alpha)*rhs.data(i,j,k)
            }
        }
    }
    resultData.clamp(0,255);
    result.data = resultData;
    return result;
}

/**
 * Green Screen
 * 
 * This function substitutes a pixel with the corresponding one in a background image 
 * if its colors are in the surrounding (+- threshold) of a given color (rgb).
 * 
 * (rgb - threshold) <= pixel <= (rgb + threshold)
 * 
 * 
 * @param bkg The second image used as background
 * @param rgb[] The color to substitute (rgb[0] = RED, rgb[1]=GREEN, rgb[2]=BLUE) 
 * @param threshold[] The threshold to add/remove for each color (threshold[0] = RED, threshold[1]=GREEN, threshold[2]=BLUE) 
 * @return returns a new DAISGram containing the result.
 */  
DAISGram DAISGram::greenscreen(DAISGram & bkg, int rgb[], float threshold[]){
    if (getRows() != bkg.data.rows() || getCols() != bkg.data.cols() || getDepth() != bkg.data.depth())
    {
        throw dimension_mismatch();
    }
    DAISGram result;
    Tensor resultData(data);
    int rgbChecklist = 0;
    for(int i = 0; i<getRows(); i++){
        for(int j = 0; j<getCols(); j++){
            for(int k = 0; k<getDepth(); k++){
                if((rgb[k] - threshold[k]) <= resultData(i,j,k) && resultData(i,j,k)<=(rgb[k] + threshold[k])){
                    rgbChecklist++;
                }
            }
            if(rgbChecklist==3){
                for(int h = 0; h<getDepth(); h++){
                    resultData(i,j,h) = bkg.data(i,j,h);
                }
            }
            rgbChecklist = 0;
        }
    }
    result.data = resultData;
    return result;
}

/**
 * Equalize
 * 
 * Stretch the distribution of colors of the image in order to use the full range of intesities.
 * 
 * See https://it.wikipedia.org/wiki/Equalizzazione_dell%27istogramma
 * 
 * @return returns a new DAISGram containing the equalized image.
 */  
DAISGram DAISGram::equalize(){
    throw method_not_implemented();
}
