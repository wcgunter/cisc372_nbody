#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"
#include <stdio.h>

extern vector3 **d_accels, *d_hPos, *d_hVel, *d_accel_sum;
//declare mass array
extern double *d_mass;

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute() {
	//each of our kernels will need different numbers of blocks and threads (thanks @prof silber!)
	//as such we will define them below.

	//we need to figure out how many blocks we need to cover all NUMENTITIES bodies in our simulation
	//for calculate accels:

	int computeAccelsBlockDim = (NUMENTITIES + 7) / 8;
	dim3 computeAccelsTPB(8, 8, 3);
	dim3 computeAccelsBlocks(computeAccelsBlockDim, computeAccelsBlockDim);

	computeAccels<<<computeAccelsBlocks, computeAccelsTPB>>>(d_accels, d_hPos, d_mass);

	//for sumCols, we can use the max 1024 threads per block and one block per column:

	int sumColsTPB = 32;
	dim3 sumColsBlocks(NUMENTITIES, 3);
	int sumColsSharedMem = sumColsTPB * sizeof(double) * 2;

	sumCols<<<sumColsBlocks, sumColsTPB, sumColsSharedMem>>>(d_accels, d_accel_sum);

	//for updatePos, we need to break up into chunks with an extra dimension of 3 (for x, y, z)
	int updatePosBlockDim = (NUMENTITIES + 7) / 8;
	dim3 updatePosTPB(8, 3);

	updatePos<<<updatePosBlockDim, updatePosTPB>>>(d_accel_sum, d_hPos, d_hVel);

	//originally had copy-back to host here but was causing lots of extra overhead when not needed
	//now we request the data back in nbody.cu

}

//computeAccels: Calculates the acceleration of each object in the system
//Parameters: d_hPos - the array of positions of the objects
//            d_accels - the array of accelerations of the objects
//            d_mass: the array of masses of the objects
//Returns: None
//Side Effect: Modifies the accels array with the new values
__global__ void computeAccels(vector3** d_accels, vector3* d_hPos, double* d_mass){
	//we need to get our thread information (indexes in x, y, z)
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z; //this represents our dimensions (x, y, z)
	//if we are going to do each dimension (k) in its own thread, we are going to need a shared array for each block to store dist.
	__shared__ vector3 shDistance[8][8];

	//we need to check that i and j are not more than NUMENTITIES
	if (i >= NUMENTITIES || j >= NUMENTITIES) return;

	//if we are on the diagonal, there is no acceleration due to itself so set to all zeroes
	if (i == j) {
		d_accels[i][j][k] = 0;
	} else {
		//we need to calculate the distance between the objects
		shDistance[threadIdx.x][threadIdx.y][k] = d_hPos[i][k] - d_hPos[j][k];

		//we need to sync threads here or we may start calculating without being done >:(
		__syncthreads();

		//the following three lines are from silber's original serial code... (with minor modifications to support parallelization of 3d)
		double magnitude_sq = shDistance[threadIdx.x][threadIdx.y][0] * shDistance[threadIdx.x][threadIdx.y][0] + shDistance[threadIdx.x][threadIdx.y][1] * shDistance[threadIdx.x][threadIdx.y][1] + shDistance[threadIdx.x][threadIdx.y][2] * shDistance[threadIdx.x][threadIdx.y][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;

		//fill the vector with the results (this is only partial because 1 thread does 1 dim)
		d_accels[i][j][k] = accelmag * shDistance[threadIdx.x][threadIdx.y][k] / magnitude;
	}
}

//sumCols: Sums the columns of the accels array and stores the result in accel_sum
//Parameters: d_accels - the array of accelerations of the objects
//            d_accel_sum - the array of accelerations of the objects
//Returns: None
//Side Effect: Modifies the accel_sum array with the new values
__global__ void sumCols(vector3** d_accels, vector3* d_accel_sum){
	//this code is based off of / derived from silber's serial code
	//we need to get our information from the thread
	int row = threadIdx.x;
	int col = blockIdx.x; //we use block index here because we want to sum the columns with each column getting 1 block
	int dim = blockIdx.y;
	extern __shared__ double shArr[];
	__shared__ int offset;
	int blocksize = blockDim.x;
	int arrSize = NUMENTITIES;
	shArr[row] = row < arrSize ? d_accels[col][row][dim] : 0;
	if (row == 0) {
		offset = blocksize;
	}
	__syncthreads();
	while (offset < arrSize) {
		shArr[row+blocksize] = row+blocksize < arrSize ? d_accels[col][row+offset][dim] : 0;
		__syncthreads();
		if (row == 0) {
			offset += blocksize;
		}
		double sum = shArr[2*row] + shArr[2*row+1];
		__syncthreads();
		shArr[row] = sum;
	}
	__syncthreads();
	for (int stride = 1; stride < blocksize; stride *= 2) {
		int arrIdx = row*stride*2;
		if (arrIdx + stride < blocksize) {
			shArr[arrIdx] += shArr[arrIdx + stride];
		}
		__syncthreads();
	}
	if (row == 0) {
		d_accel_sum[col][dim] = shArr[0];
	}
}

//updatePos: Updates the positions and velocities of the objects in the system based on accel_sum and position and velocity.
//Parameters: hPos - the array of positions of the objects
//            hVel - the array of velocities of the objects
//            accel_sum - the array of accelerations of the objects
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
__global__ void updatePos(vector3* accel_sum, vector3* hPos, vector3* hVel) {
	//we will launch a thread for each index in accel_sum so no need for for loop here
	//get our thread information
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y;

	//we need to check that the thread index is not greater than number of entities
	if (i >= NUMENTITIES) return;

	hVel[i][k] += accel_sum[i][k] * INTERVAL;
	hPos[i][k] = hVel[i][k] * INTERVAL;
}
