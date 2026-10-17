#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <cassert>
#include <map>
#include <set>
#include <chrono>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <time.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#define u64 unsigned long long

using namespace std;

#define MAXN 66
#define MAXM 960

#ifndef RUNTIME
#define RUNTIME 65535
#endif

//Name of the inputfile, outputfile, and logfile
char IFILENAME[160],OUTFILENAME[160],LOGFILENAME[160];
int REC=0; //running count
time_t RUNID; //running ID (extracted from the last 6 digits of the system timestamp)

//Store the g-Xor sequence and length of g-Xor sequence
struct Program{int t,a,b,y;}programSeq[MAXM];
int lenProg=0;

//Store the distance after adding the new variable B[a]^B[b]
struct Metrics {
    int a, b; //Two candidate index
    int ind; //Pair index
    int d1;  //sum of distance
    int d2;  //sum of (square of distance)
    bool operator<(const Metrics& other) const {
        if (d1 != other.d1) {
            return d1 < other.d1; 
        }
        return d2 > other.d2;
    }
};

//Comparator for thrust::sort (but it's actually unnecessary, because the number of candidates won't exceed 10**6.)
struct MetricsComparator {
    __host__ __device__
    bool operator()(const Metrics& x, const Metrics& y) const {
        if (x.d1 != y.d1) {
            return x.d1 < y.d1;
        }
        return x.d2 > y.d2;
    }
};


u64 *d_C,*C; //C: Store the binomial coefficient at the gpu side and the cpu side. It will not be modified after initial calculation
u64 Y[MAXN],*d_Y; //Y: Store the targets both at the cpu and the gpu side
u64 B[MAXM],*d_B,*d_nextB; //B: current variables (bases) at cpu and gpu side. nextB: C(m,2) candidates at gpu side.
int D[MAXN],*d_D,*d_ND; //D: store the distances at cpu and gpu side. d_ND: store the values of the new distance vector after inserting each of the C(m,2) candidates into the current base (length: C(m,2)*n)

int ND[MAXN*MAXM*MAXM]; //for debug, no use.
int N,M; //matrix dimension and current variables.
u64 *d_W[9]={nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr},**d_Wptr; //Stores the XOR sum of the results from all base combinations
u64 Wlen[9]={0,0,0,0,0,0,0,0,0},*d_Wlen; //Stores the length of the W arrays
Metrics E[MAXM*MAXM],*d_E; //Store the struct of Metrics at GPU and CPU side.
char *d_R,*d_S; //Stores the results of binary search (initialized to 0, length: C(m,2); set to 1 if the search is successful) d_R for updating distance, d_S for check the effective of the candidates. they have the same function.
unsigned *d_OK; //Store whether each candidate meets the algorithm requirements (exEasymove in BP, and the minimum non-zero value in the distance vector in A1A2).
u64 globalAvailPairs; //number of pairs that meet the requirement

void initDealing()
{
    //Initialize variables at the beginning of each call of dealBP10()
	M=N;
	for(int i=0;i<N;i++) B[i]=((1ULL)<<i);
	for(int i=0;i<N;i++) D[i]=__builtin_popcountll(Y[i])-1;
	cudaMemcpy(d_B,B,sizeof(u64)*MAXM,cudaMemcpyHostToDevice);
	cudaMemcpy(d_D,D,sizeof(int)*MAXN,cudaMemcpyHostToDevice);
	lenProg=0;
}

__global__ void getCombArrKernel2(u64* result,u64* base,const u64* combTable,int n,int m,u64 totalComb,int tableCols=9)
{
    unsigned long long k=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=totalComb)return;
    int s=0;
    int curM=m;
    unsigned long long curK=k;
    u64 combVec=0ULL;//calculate the k-th comb-seq value of C(n,m).
    //e.g. for C(4,2)=6, the 0th comb-seq is (0,1), and the following is (0,2),(0,3),(1,2),(1,3),(2,3)
    for(int idx=0;idx<m;++idx)
    {
        int x=s;
        while(true)
        {
            int remaining=n-x-1;
            unsigned long long numComb=0;

            if(curM-1<=0)
                numComb=1;
            
            else if(remaining>=(curM-1))
                numComb=combTable[remaining*tableCols+(curM-1)];
            else
                numComb=0;
            if(curK<numComb)
            {
                combVec^=base[x];
                s=x+1;
                curM--;
                break;
            }
            else
            {
                curK-=numComb;
                x++;
            }
        }
    }
    result[k]=combVec;
}

void combEntry2(int baseNumber,int combNumber) //combNumber should be in {1,2,3,4,5,6,7,8}
{
    //calculate all the C(baseNumber, combNumber) combination results
	Wlen[combNumber]=C[baseNumber*9+combNumber];
	if(d_W[combNumber]==nullptr) cudaMalloc(&d_W[combNumber],8ULL*C[baseNumber*9+combNumber]);
    int blockSize=256; //can be 512
    unsigned long long gridSize=(C[baseNumber*9+combNumber]+blockSize-1) / blockSize;
    getCombArrKernel2<<<gridSize, blockSize>>>(d_W[combNumber],d_B,d_C,baseNumber,combNumber,C[baseNumber*9+combNumber]);
	thrust::sort(thrust::device,d_W[combNumber],d_W[combNumber]+Wlen[combNumber]); //sort here.
}
void newDistancePre()
{
    //Update the W array in each iteration
	for(int i=0;i<9;i++)
	{
		if(d_W[i]!=nullptr)
		{
			cudaFree(d_W[i]);
			d_W[i]=nullptr;
			Wlen[i]=0;
		}
	}
	int dmax=0;
	for(int i=0;i<N;i++)
		if(D[i]>dmax) dmax=D[i];
	int total=(dmax)/2;
	printf("DMAX=%d,TOTAL=%d\n",dmax,total);

	for(int i=1;i<=total;i++)
	{
		Wlen[i]=C[9*M+i];
		cudaMalloc(&d_W[i],8ULL*Wlen[i]);
		combEntry2(M,i);
	}
	cudaMemcpy(d_D,D,sizeof(int)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Wlen,Wlen,sizeof(Wlen),cudaMemcpyHostToDevice);
	cudaMemcpy(d_Wptr,d_W,9*sizeof(u64*),cudaMemcpyHostToDevice);
}

__global__ void getCandidateKernel(u64* CandidateBx,const u64* Bx,int mx,long long totalPairs,Metrics *Ex)
{
    //Parallel compute all C(m,2) candidates at gpu side.
    long long tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<totalPairs)
    {
        //from count and n to calculate i and j
        /*
            for(int i=0;i<n;i++)
                for(int j=i+1;j<n;j++)
                    cout<<count++<<endl;
        */
        double m=mx;
        int i=(int)((2.0*m-1.0-sqrt((2.0*m-1.0)*(2.0*m-1.0)-8.0*tid))/2.0);
        long long rowStartIdx=1LL*i*mx-1LL*i*(i+1)/2;
        int j=(int)(tid-rowStartIdx+i+1);
        CandidateBx[tid]=Bx[i]^Bx[j];
        Ex[tid].a=i; //store the i and j value first
        Ex[tid].b=j;
    }
}

void getCandidateEntry()
{
    //Entry of getCandidateKernel
    u64 totalPairs=1LL*M*(M-1)/2;
    if (totalPairs<=0) return;
    int blockSize=256;
    long long gridSize=(totalPairs+blockSize-1)/blockSize;
    getCandidateKernel<<<(int)gridSize,blockSize>>>(d_nextB,d_B,M,totalPairs,d_E);
    // cudaDeviceSynchronize();
}


__device__ char deviceBinSearch(const u64* arr,u64 len,u64 target)
{
    //binary search at gpu side.
    if(len==0) return 0;
    long long L=0,R=(long long)len-1;
    while(L<=R)
    {
        long long M=L+(R-L)/2;
        if(arr[M]==target) return 1;
        if(arr[M]<target) L=M+1;
        else R=M-1;
    }
    return 0;
}

__global__ void updDistance4Dis2(char *resultArray,
    const u64 Yxgpu,
    const u64 *CandidateB,
    const u64 *sortedBase,
    const u64 totalTasks,
    const u64 totalComb,
    const u64 totalPairs,
    unsigned *tagOKarr=nullptr,
    const u64 m=0
    )
{
    /*
    NOTE:
    This kernel is used for initial filtering of candidate variables.
    When the current distance D[i] == 2, it attempts to find if a candidate 
    variable XORed with one base variable from the library equals the target.
    Due to XOR properties, we check if Tx is 0 directly without binary search.
    */
    u64 tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=totalTasks) return;
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    u64 Tx;
    Tx=(Yxgpu^CandidateB[pairID]^sortedBase[combID]);
    if(!Tx)
    {
        resultArray[pairID]=1; //tag found.
        if(m&&tagOKarr!=nullptr) 
        {
            double mf=(double)m;
            int i=(int)((2.0*mf-1.0-sqrt((2.0*mf-1.0)*(2.0*mf-1.0)-8.0*pairID))/2.0);
            long long rowStartIdx=1LL*i*m-1LL*i*(i+1)/2;
            int j=(int)(pairID-rowStartIdx+i+1);
            tagOKarr[pairID]=((i<<16)|(j&0xffff)); // Encode i and j into tagOKarr for later stages
        }
    }
    
}

__global__ void updDistance4Dis3Plus(char *resultArray,
    const u64 runOffset,
    const u64 Yxgpu,
    const u64 *CandidateB,
    const u64 *w_k1,
    const u64 totalComb_1,
    const u64 *w_k2,   
    const u64 totalComb_2, 
    const u64 totalTasks,
    const u64 totalPairs,
    unsigned *tagOKarr=nullptr, //For A1 Check
    const int m=0 //For A1 Check
    )
{
    /*
    NOTE:
    Generic filtering for cases where D[i] >= 3.
    Uses typical MITM strategy: splitting distance into K1 and K2.
    Iterates through W[K2] and performs deviceBinSearch on W[K1].
    */
    u64 tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=totalTasks) return;
    tid+=runOffset;
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    u64 Tx=Yxgpu^CandidateB[pairID]^w_k2[combID];

    char r=deviceBinSearch(w_k1,totalComb_1,Tx); //binarysearch at GPU.
    if(r)
    {
        resultArray[pairID]=1; //tag found
        if(tagOKarr&&m) 
        {
            double mf=(double)m;
            int i=(int)((2.0*mf-1.0-sqrt((2.0*mf-1.0)*(2.0*mf-1.0)-8.0*pairID))/2.0);
            long long rowStartIdx=1LL*i*m-1LL*i*(i+1)/2;
            int j=(int)(pairID-rowStartIdx+i+1);
            tagOKarr[pairID]=((i<<16)|(j&0xffff));
        }
    }
}



__global__ void updDistance4Dis2A1(char *resultArray,
    const u64 Yxgpu,
    const u64 *Bases,
    const u64 *sortedBase,
    const u64 totalTasks,
    const u64 totalComb,
    const u64 totalPairs,
    unsigned *tagOKarr=nullptr,
    const u64 m=0
    )
{
    /*
    NOTE:
    A1 version: Designed for candidates already filtered by checkCandidateEntry.
    Instead of traversing CandidateB, it decodes i and j from tagOKarr and retrieves 
    values from the original Bases array. Used in computeAllDistanceEntry to 
    calculate actual values in the ND matrix.
    */
    u64 tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=totalTasks) return;
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    int pairInd=tagOKarr[pairID]; // Retrieve encoded indices from filtered candidate pool
    u64 Tx=(Yxgpu^Bases[pairInd&0xffff]^Bases[pairInd>>16]^sortedBase[combID]);
 
    if(!Tx)
    {
        resultArray[pairID]=1;
    }
    
}

__global__ void updDistance4Dis3PlusA1(char *resultArray,
    const u64 runOffset,
    const u64 Yxgpu,
    const u64 *Bases,
    const u64 *w_k1,
    const u64 totalComb_1,
    const u64 *w_k2,   
    const u64 totalComb_2, 
    const u64 totalTasks,
    const u64 totalPairs,
    unsigned *tagOKarr //For A1 Check
    )
{
    /*
    NOTE:
    A1 version: Handles the D[i] >= 3 case for filtered candidates.
    Similarly retrieves specific (i, j) pairs via tagOKarr to avoid storing 
    massive CandidateB matrices or redundant XOR calculations.
    */
    u64 tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=totalTasks) return;
    tid+=runOffset;
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    int pairInd=tagOKarr[pairID];
    u64 Tx=Yxgpu^Bases[pairInd&0xffff]^Bases[pairInd>>16]^w_k2[combID];
    char r=deviceBinSearch(w_k1,totalComb_1,Tx);
    if(r)
        resultArray[pairID]=1;
}


void calculateDis4ezMoveEntry(u64 newBase, int a)
{
    //Deal with distance update in the scenario of easymove
    char R_tmp[MAXN];
    cudaMemset(d_R,0,N);
    D[a]=0;
    for(int i=0;i<N;i++)
    {
        if(D[i]==0||D[i]==1) continue; //Note: Assume Y is full ranked, other D[i]==1 will note got to be 0.
        if(D[i]==2)
        {
            for(int j=0;j<M;j++)
            {
                if(!(Y[i]^newBase^B[j]))
                {
                    D[i]=1;
                    break;
                }
            }
        }
        if(D[i]>=3)
        {
            int K=D[i]-1;
            int K2=K/2;
            int K1=K-K2;
            u64 totalTasks=1*Wlen[K2];
            int threadsPerBlock=256;
            u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
            cudaMemcpy(d_nextB,&newBase,sizeof(u64),cudaMemcpyHostToDevice);
            updDistance4Dis3Plus<<<gridSize,threadsPerBlock>>>(d_R+i,0,Y[i],d_nextB,d_W[K1],Wlen[K1],d_W[K2],Wlen[K2],totalTasks,1);
        }
    }
    cudaMemcpy(R_tmp,d_R,N,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
        D[i]=D[i]-R_tmp[i];
    cudaMemcpy(d_D,D,sizeof(int)*N,cudaMemcpyHostToDevice);
}

int easyMove()
{
    //Deal with easymove
	int a,found=0;
	for(a=0;a<N;a++)
		if(D[a]==1)
		{
			found=1;
			break;
		}
	if(!found) return 0;
	for(int i=0;i<M;i++)
		for(int j=1+i;j<M;j++)
		{
			if(Y[a]==(B[i]^B[j]))
			{
                // printf("%016llx %016llx %016llx %d\n",Y[a],B[i],B[j],Y[a]==(B[i]^B[j]));
                u64 newBase=B[i]^B[j];
                cudaMemcpy(d_D+a, D+a, sizeof(int), cudaMemcpyHostToDevice);
                calculateDis4ezMoveEntry(newBase,a);
				programSeq[lenProg].t=M;
				programSeq[lenProg].a=i;
				programSeq[lenProg].b=j;
				programSeq[lenProg].y=a;
				lenProg++;
				B[M]=B[i]^B[j];
				cudaMemcpy(&d_B[M],&B[M],sizeof(u64),cudaMemcpyHostToDevice);
				printf("%d %d %d %d\n",M,i,j,a);
                M++;
				return 1;
			}
		}
    puts("Some Error Occurs!");
    exit(2);
    return 0;
}
__global__ void calculateND4Yi(int *NDgpu,char *resultArr,u64 totalPairs,int targetID,int curDis,int Nx)
{
    u64 pairID=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(pairID>=totalPairs) return;
    if(curDis<=1) NDgpu[pairID*Nx+targetID]=curDis;
    else NDgpu[pairID*Nx+targetID]=curDis-resultArr[pairID];
}
void computeAllDistanceEntry()
{
    u64 totalPairs = globalAvailPairs;
    for(int i=0;i<N;i++)
    {
        int Di=D[i];
        if(Di==2)
        {
            cudaMemset(d_R,0,totalPairs);
            u64 totalTasks=totalPairs*M;
            int threadsPerBlock = 256;
            u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
            updDistance4Dis2A1<<<gridSize,threadsPerBlock>>>(d_R,Y[i],d_B,d_B,totalTasks,M,totalPairs,d_OK);
        }
        if(Di>=3)
        {
            cudaMemset(d_R,0,totalPairs);
            int K=Di-1;
            int K2=K/2;
            int K1=K-K2;
            u64 remainingTasks=totalPairs*Wlen[K2];
            u64 runOffset=0;
            while(remainingTasks)
            {
                u64 totalTasks=(remainingTasks>256ULL*0x7fffff00ULL)?256ULL*0x7fffff00ULL:remainingTasks;
                int threadsPerBlock=256;
                u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
                updDistance4Dis3PlusA1<<<gridSize,threadsPerBlock>>>(d_R,runOffset,Y[i],d_B,d_W[K1],Wlen[K1],d_W[K2],Wlen[K2],totalTasks,totalPairs,d_OK);
                cudaDeviceSynchronize();
                runOffset+=totalTasks;
                remainingTasks-=totalTasks;
            }
        }        
        u64 totalTasks=totalPairs*1;
        int threadsPerBlock=256;
        u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
        calculateND4Yi<<<gridSize,threadsPerBlock>>>(d_ND,d_R,totalTasks,i,Di,N);
    }
}

u64 checkCandidateEntry()
{
    int minDis=99999999,minDisCount=0,minDisTargets[MAXN];
    for(int i=0;i<N;i++)
    {
        if(!D[i]) continue;
        if(D[i]<minDis)
        {
            minDis=D[i];
            minDisCount=1;
            minDisTargets[0]=i;
        }
        else if(D[i]==minDis)
            minDisTargets[minDisCount++]=i;
    }
    u64 totalPairs=1LL*M*(M-1)/2;
    cudaMemset(d_S,0,totalPairs);
    cudaMemset(d_OK,-1,sizeof(unsigned)*totalPairs);
    if(minDis==2)
    {
        for(int i=0;i<minDisCount;i++)
        {
            u64 totalTasks=totalPairs*M;
            int threadsPerBlock = 256;
            u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
            updDistance4Dis2<<<gridSize,threadsPerBlock>>>(d_S,Y[minDisTargets[i]],d_nextB,d_B,totalTasks,M,totalPairs,d_OK,M);
        }
    }
    else
    {
        for(int i=0;i<minDisCount;i++)
        {
            int K=minDis-1;
            int K2=K/2;
            int K1=K-K2;
            u64 remainingTasks=totalPairs*Wlen[K2];
            u64 runOffset=0;
            while (remainingTasks)
            {
                u64 totalTasks=(remainingTasks>256ULL*0x7fffff00ULL)?256ULL*0x7fffff00ULL:remainingTasks;
                int threadsPerBlock=256;
                u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
                updDistance4Dis3Plus<<<gridSize,threadsPerBlock>>>(d_S,runOffset,Y[minDisTargets[i]],d_nextB,d_W[K1],Wlen[K1],d_W[K2],Wlen[K2],totalTasks,totalPairs,d_OK,M);
                runOffset+=totalTasks;
                remainingTasks-=totalTasks;
            }            
        }
    }
    
    thrust::sort(thrust::device,d_OK,d_OK+totalPairs);
    unsigned* bound_ptr = thrust::lower_bound(thrust::device, d_OK, d_OK + totalPairs, 0x70000000U);
    u64 count = bound_ptr - d_OK;
    return count;
}


__global__ void calcuteMetricsKernel(Metrics* d_E,const int* d_ND,int n,int m,long long totalPairs,unsigned *d_OK=nullptr)
{
    long long k=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=totalPairs) return;

    int i,j;
    if (d_OK==nullptr&&totalPairs==(1LL*m*(m-1)/2))
    {
        double mf=(double)m;
        i=(int)((2.0*mf-1.0-sqrt((2.0*mf-1.0)*(2.0*mf-1.0)-8.0*k))/2.0);
        long long rowStartIdx=1LL*i*m-1LL*i*(i+1)/2;
        j=(int)(k-rowStartIdx+i+1);
    }
    else
    {
        unsigned tmp=d_OK[k];
        i=(tmp>>16);
        j=(tmp&0xffff);
    }
    int sumD=0;
    int sumN=0;
    const int* current_ND_row=&d_ND[k*n];

    for (int tid=0;tid<n;tid++){
        int val=current_ND_row[tid];
        sumD+=val;
        sumN+=val*val;
    }
    d_E[k].a=i;
    d_E[k].b=j;
    d_E[k].ind=k;
    d_E[k].d1=sumD;
    d_E[k].d2=sumN;
}
int calcuteMetricsEntry()
{
    u64 totalPairs=globalAvailPairs;
    int blockSize=256;
    u64 gridSize=(totalPairs+blockSize-1)/blockSize;

    calcuteMetricsKernel<<< (int)gridSize,blockSize >>>(d_E,d_ND,N,M,totalPairs,d_OK);

    cudaMemcpy(E,d_E,sizeof(Metrics)*totalPairs,cudaMemcpyDeviceToHost);
    sort(E,E+totalPairs);
    int k=0;
    if (totalPairs>0)
    {
        Metrics target=E[0];
        // Use binary search to find the boundary: find the last index equal to target
        int L=0,R=totalPairs-1;
        int ans=0;
        while (L<=R)
        {
            long long mid=L+(R-L)/2;
            if (E[mid].d1==target.d1&&E[mid].d2==target.d2)
            {
                ans=mid;    
                L=mid+1;   
            } 
            else
                R=mid-1;   
        }
        k=ans+1; 
    }
    return k;
}

void dealBP10()
{
    sprintf(OUTFILENAME,"Record%s-%d-%06x.ans",IFILENAME,++REC,RUNID);
	sprintf(LOGFILENAME,"Record%s-%d-%06x.log",IFILENAME,REC,RUNID);
	unsigned long long TotalTimeCost=0;
	FILE *fp,*flp;
    auto T1=chrono::high_resolution_clock::now(),T2=chrono::high_resolution_clock::now();
	initDealing();
	while(1)
	{
        flp=fopen(LOGFILENAME,"a");
        T2=chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<std::chrono::milliseconds>(T2 - T1);
        printf("Last Time Cost: %08llu ms\n",duration.count());
		fprintf(flp,"Last Time Cost: %08llu ms\n",duration.count());
        TotalTimeCost+=duration.count();
		printf("Total Time Cost: %08llu ms\n",TotalTimeCost);
		fprintf(flp,"Total Time Cost: %08llu ms\n",TotalTimeCost);
        T1=T2;

		newDistancePre();
        printf("culcated dis.");

        fp=fopen(OUTFILENAME,"a");


		int sumDist=0,sumNorm=0;
		for(int i=0;i<N;i++)
		{
			sumDist+=D[i];
			sumNorm+=D[i]*D[i];
		}
        printf("Total Distance %d: ",sumDist);
        fprintf(flp,"Total Distance %d: ",sumDist);
        
        for(int i=0;i<N;i++)
        {
            printf("%d%c",D[i],32-22*(i+1==N));
            fprintf(flp,"%d%c",D[i],32-22*(i+1==N));
        }
        fclose(flp);
        if(!sumDist) break;
		if(easyMove())
		{
            Program tmp=programSeq[lenProg-1];
            fprintf(fp,"%d %d %d %d\n",tmp.t,tmp.a,tmp.b,tmp.y);
            fclose(fp);
			continue;
		}

		getCandidateEntry();

        globalAvailPairs=checkCandidateEntry();
        cout<<globalAvailPairs<<endl;

        computeAllDistanceEntry();
        
		int bestCount=calcuteMetricsEntry();
        Metrics selected=E[rand()%bestCount];
        int besti=selected.a,bestj=selected.b;
        int bestOffset=selected.ind;
        cudaMemcpy(D,d_ND+(N*bestOffset),sizeof(int)*N,cudaMemcpyDeviceToHost);
        programSeq[lenProg].t=M;
        programSeq[lenProg].a=besti;
        programSeq[lenProg].b=bestj;
        programSeq[lenProg].y=-1; //Should not meet a target here...
        lenProg++;
        B[M]=B[besti]^B[bestj];
        cudaMemcpy(d_B+M,B+M,8,cudaMemcpyHostToDevice); 
        fprintf(fp,"%d %d %d %d\n",M,besti,bestj,-1);
        fclose(fp);
        M++;
	}
}
void initGlobals()
{
	C=(u64*)malloc(9LL*sizeof(u64)*MAXM);
	C[0]=1;
    for(int i=0;i<MAXM;i++)
        for(int j=0;j<=i&&j<9;j++)
        {
            if(j==0||j==i) C[i*9+j]=1;
            else
                C[i*9+j]=C[(i-1)*9+(j-1)]+C[(i-1)*9+j];
        }
	cudaMalloc(&d_C,9LL*sizeof(u64)*MAXM);
	cudaMemcpy(d_C,C,MAXM*72LL,cudaMemcpyHostToDevice);
	cudaMalloc(&d_B,sizeof(u64)*MAXM);
	cudaMalloc(&d_nextB,sizeof(u64)*MAXM*MAXM);
	cudaMalloc(&d_Y,sizeof(u64)*MAXN);
	cudaMemcpy(d_Y,Y,sizeof(u64)*MAXN,cudaMemcpyHostToDevice);
	cudaMalloc(&d_D,sizeof(int)*MAXN);
	cudaMalloc(&d_ND,sizeof(int)*MAXN*MAXM*MAXM);
	cudaMalloc(&d_Wlen,sizeof(u64)*MAXN*MAXM*MAXM);
	cudaMalloc(&d_Wptr,sizeof(u64*)*9);
	cudaMalloc(&d_E,sizeof(Metrics)*MAXM*MAXM);
    cudaMalloc(&d_R,sizeof(char)*MAXM*MAXM);
    cudaMalloc(&d_S,sizeof(char)*MAXM*MAXM);
    cudaMalloc(&d_OK,sizeof(unsigned)*MAXM*MAXM);
}

int main(int argc, char *argv[])
{
    time(&RUNID);
    RUNID&=0xffffff;
    srand(RUNID);
	FILE *fp;
    if(argc==1)
	{
		fp=stdin;
        IFILENAME[0]=0;
	}
	else
	{
		fp=fopen(argv[1],"r");
		sprintf(IFILENAME,"%s",argv[1]);
	}
	printf("%p\n",fp);
	fscanf(fp,"%d",&N);
	M=N;
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			int x;
			fscanf(fp,"%d",&x);
			if(x) Y[i]|=(1ULL<<j);
		}
	}
	initGlobals();
    unsigned short runtime=RUNTIME;
	while(runtime--) dealBP10();
}