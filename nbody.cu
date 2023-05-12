#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
vector3 **d_accels, *d_accel_sum;
vector3* temp[NUMENTITIES];
double *mass, *d_mass;

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//initDeviceMemory: Create storage for numObjects entities in our system on the GPU
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the for the device variables on the GPU

//I was going to do this in a function, but it causes errors on Darwin (and only on darwin?)
/*
void initDeviceMemory() {
	cudaMalloc((void**)&d_accels, NUMENTITIES * sizeof(vector3*));

	//2d arrays are funky in cuda, here's a workaround
	for (int i = 0; i < NUMENTITIES; i++) {
		cudaMalloc(&temp[i], sizeof(vector3) * NUMENTITIES);
	}
	cudaMemcpy(d_accels, temp, NUMENTITIES * sizeof(vector3*), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_hPos, NUMENTITIES*sizeof(vector3));

	cudaMalloc((void**)&d_hVel, NUMENTITIES*sizeof(vector3));

	cudaMalloc((void**)&d_accel_sum, NUMENTITIES*sizeof(vector3));

	cudaMalloc((void**)&d_mass, NUMENTITIES*sizeof(double));
}
*/

//freeDeviceMemory: Free storage allocated by a previous call to initDeviceMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to device global variables
void freeDeviceMemory() {
	for (int i = 0; i < NUMENTITIES; i++) {
		cudaFree(temp[i]);
	}

	cudaFree(d_accels);

	cudaFree(d_hPos);

	cudaFree(d_hVel);

	cudaFree(d_accel_sum);
	
	cudaFree(d_mass);
}


//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j= start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);

	//The cuda malloc statements are here instead of the function because darwin breaks if its in a function :))))))))))

	cudaMalloc((void**)&d_accels, NUMENTITIES * sizeof(vector3*));

	//2d arrays are funky in cuda, here's a workaround
	for (int i = 0; i < NUMENTITIES; i++) {
		cudaMalloc(&temp[i], sizeof(vector3) * NUMENTITIES);
	}
	cudaMemcpy(d_accels, temp, NUMENTITIES * sizeof(vector3*), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_hPos, NUMENTITIES*sizeof(vector3));

	cudaMalloc((void**)&d_hVel, NUMENTITIES*sizeof(vector3));

	cudaMalloc((void**)&d_accel_sum, NUMENTITIES*sizeof(vector3));

	cudaMalloc((void**)&d_mass, NUMENTITIES*sizeof(double));

	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif

	//we need to send the system to the GPU
	cudaMemcpy(d_hPos, hPos, NUMENTITIES*sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, NUMENTITIES*sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, NUMENTITIES*sizeof(double), cudaMemcpyHostToDevice);

	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}

	cudaMemcpy(hPos, d_hPos, NUMENTITIES*sizeof(vector3), cudaMemcpyDeviceToHost);

	cudaMemcpy(hVel, d_hVel, NUMENTITIES*sizeof(vector3), cudaMemcpyDeviceToHost);

	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
	freeDeviceMemory();
}
