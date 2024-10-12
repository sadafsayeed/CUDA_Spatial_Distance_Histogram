/* ==================================================================
	Programmer: Sadaf Sayeed Chowdhury (chowdhury25@usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the GAIVI machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* Thesea are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/

// I modified this function to print any histogram that is passed to it
void output_histogram(bucket* histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

// This is my kernel function
__global__ void p2p_dist_parallel_computing_kernel (atom* atoms, bucket* histogram, int PDH_acnt, double PDH_res){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<PDH_acnt){
		for(int j=i+1; j<PDH_acnt; j++){
			double dist_x = atoms[i].x_pos - atoms[j].x_pos;
            double dist_y = atoms[i].y_pos - atoms[j].y_pos;
            double dist_z = atoms[i].z_pos - atoms[j].z_pos;
            double dist = sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

            int h_pos = (int)(dist / PDH_res);
			atomicAdd(&(histogram[h_pos].d_cnt), 1);
		}
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	
	atom* atom_list_d;
	bucket* histogram_gpu;
	bucket* histogram_d;

	histogram_gpu = (bucket *)malloc(sizeof(bucket)*num_buckets);
	cudaMalloc(&atom_list_d,sizeof(atom)*PDH_acnt);
	cudaMalloc(&histogram_d,sizeof(bucket)*num_buckets);
	cudaMemcpy(atom_list_d,atom_list,sizeof(atom)*PDH_acnt,cudaMemcpyHostToDevice);
    cudaMemcpy(histogram_d,histogram,sizeof(bucket)*num_buckets,cudaMemcpyHostToDevice);

	// This is for measuring the GPU's time
	cudaEvent_t start_event, end_event;
	float gpu_time;
	cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
	cudaEventRecord(start_event);

	dim3 numThreadsPerBlock(512);
	dim3 numBlocks((PDH_acnt + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x);
	
	// This calls the kernel
	p2p_dist_parallel_computing_kernel<<<numBlocks, numThreadsPerBlock>>>(atom_list_d,histogram_d,PDH_acnt,PDH_res);
	
	cudaEventRecord(end_event);
	cudaEventSynchronize(end_event);
	cudaEventElapsedTime(&gpu_time, start_event, end_event);
	
	cudaMemcpy(histogram_gpu,histogram_d,sizeof(bucket)*num_buckets,cudaMemcpyDeviceToHost);
	
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* print out the histogram */
	printf("\nWith CPU:\n");
	report_running_time();
	output_histogram(histogram);
	
	printf("\nWith GPU:\n");
	printf("Running time for GPU version: %f\n", gpu_time/1000.0); 
	output_histogram(histogram_gpu);

	bucket *histogram_diff = (bucket *)malloc(sizeof(bucket) * num_buckets);

	int diff_found = 0;  

	// This is for comparing the two histograms produced by the cpu and the gpu
	for (int i = 0; i < num_buckets; i++) {
		histogram_diff[i].d_cnt = histogram[i].d_cnt - histogram_gpu[i].d_cnt;
		if (histogram_diff[i].d_cnt != 0) {
			diff_found = 1;  
		}
	}

	// If there is any discrepancy between the buckets, the difference gets printed or else nothing prints
	if (diff_found) {
		printf("\nComparing CPU to GPU:\n");
		output_histogram(histogram_diff);
	}

	cudaFree(atom_list_d);
	cudaFree(histogram_d);
	
	return 0;
}