
/*
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)

Modified by Jordan Bonilla and Matthew Cedeno (2016)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
#include "ta_utilities.hpp"

#define PI 3.14159265358979


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

// Performs a high pass filter on the sinogram data
__global__ void cudaHighPassKernel(cufftComplex *raw_data, const int sinogram_width, const int size){
  uint idx = blockDim.x * blockIdx.x + threadIdx.x;

  // The scaling factor is 0 at sinogram_width / 2, and scales to 1 at 0 and
  // sinogram width, so we first shift down sinogram_width / 2, putting the
  // center at 0. This means the scaling factor is then abs(shifted value) /
  // (sinogram_width / 2)
  float scalingFactor = (idx % sinogram_width) - sinogram_width / 2.0;
  scalingFactor = abs(scalingFactor)/(sinogram_width / 2.0);

  // Scale every value by the scaling factor
  while(idx < size){
    raw_data[idx].x = raw_data[idx].x * scalingFactor;
    raw_data[idx].y = raw_data[idx].y * scalingFactor;
    idx += blockDim.x * gridDim.x;
  }
}

// Calls the high pass kernel
void cudaCallHighPassKernel(const unsigned int blocks, const unsigned int threadsPerBlock,
cufftComplex *raw_data, const int sinogram_width, const int size){
  cudaHighPassKernel<<<blocks, threadsPerBlock>>>(raw_data, sinogram_width, size);
}

__global__
// Just takes the real value of every complex sinogram data and puts it into the
// output.
void cudaCmplxToFloat(const cufftComplex *raw_data, float *output_data,
int size){
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;

    while(idx < size){
      output_data[idx] = raw_data[idx].x;
      idx += blockDim.x * gridDim.x;
    }
}

// Calls the complex to float kernel
void cudaCallCmplxToFloat(unsigned int blocks, unsigned int threadsPerBlock,
const cufftComplex *raw_data, float *output_data, int size){
  cudaCmplxToFloat<<<blocks, threadsPerBlock>>>(raw_data, output_data, size);
}

__global__
// Performs the backprojection of the image to reconstruct the sinogram
void cudaBackprojection(const float *input_data, float *output_data,
  const int sinogram_width, const int height, const int angles,
  const int size){
  uint idx = threadIdx.x + blockIdx.x * blockDim.x;
  while(idx < size){
    // Calculate the geometric location of our current pixel
    int geo_x = (idx % height) - height / 2;
    int geo_y = -1 * ((int)(idx / height)) + height / 2;
    // For each angle, check if it is an edge case. Otherwise, calculate the
    // distance we are from the center of the angle's emitter. Then find the
    // value of the sinogram for that pixel.
    for(int i = 0; i < angles; i++){
      float theta = i * PI / angles;
      int d;
      if(theta == 0){
        d = geo_x;
      }
      else if(theta == PI / 2){
        d = geo_y;
      }
      else{
        float m = -1 * (cos(theta) / sin(theta));
        float q = -1 / m;

        float x_i = (geo_y - m * geo_x) / (q - m);
        float y_i = q * x_i;
        d = (int) sqrt(pow(x_i, 2) + pow(y_i, 2));
        if((q > 0 && x_i < 0) || (q < 0 && x_i > 0)){
          d *= -1;
        }
      }
      output_data[idx] += input_data[sinogram_width * d + i];
    }
    idx += blockDim.x * gridDim.x;
  }
}

void cudaCallBackprojection(unsigned int blocks, unsigned int threadsPerBlock,
const float *input_data, float *output_data, const int sinogram_width,
const int height, const int angles, const int size){
  cudaBackprojection<<<blocks, threadsPerBlock>>>(input_data, output_data,
    sinogram_width, height, angles, size);
}

int main(int argc, char** argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Begin timer and check for the correct number of inputs
    time_t start = clock();
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input sinogram text file's name > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output text file's name >\n");
        exit(EXIT_FAILURE);
    }

    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float;
    float* dev_output;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    gpuErrchk(cudaMalloc((void**) &dev_sinogram_float, sizeof(float) * size_result));
    gpuErrchk(cudaMalloc((void**) &dev_sinogram_cmplx, sizeof(cufftComplex) * sinogram_width * nAngles));
    gpuErrchk(cudaMalloc((void**) &dev_output, sizeof(float) * size_result));

    gpuErrchk(cudaMemcpy(dev_sinogram_cmplx, sinogram_host, sizeof(cufftComplex) * sinogram_width * nAngles, cudaMemcpyHostToDevice));

    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */

    cufftHandle plan;
    cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, nAngles);
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);

    cudaCallHighPassKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx, sinogram_width, sinogram_width * nAngles);

    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);

    cudaCallCmplxToFloat(nBlocks, threadsPerBlock, dev_sinogram_cmplx, dev_sinogram_float,
    sinogram_width * height);
    
    cudaFree(dev_sinogram_cmplx);
    cufftDestroy(plan);

    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image. got it
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */

    cudaCallBackprojection(nBlocks, threadsPerBlock, dev_sinogram_float, dev_output,
    sinogram_width, height, nAngles, size_result);
    fprintf(stderr, "backproject");
    gpuErrchk(cudaMemcpy(output_host, dev_output, sizeof(float) * size_result, cudaMemcpyDeviceToHost));
    fprintf(stderr, "copying back");
    cudaFree(dev_sinogram_float);
    cudaFree(dev_output);

    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);
    printf("CT reconstruction complete. Total run time: %f seconds\n", (float) (clock() - start) / 1000.0);
    return 0;
}
