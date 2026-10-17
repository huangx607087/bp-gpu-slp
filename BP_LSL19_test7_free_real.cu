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
// #include <thrust/device_vector.h>
#define u64 unsigned long long

using namespace std;

#define MAXN 66
#define MAXM 6500

#ifndef RUNTIME
#define RUNTIME 1
#endif

#ifndef MAXDEPTH
#define MAXDEPTH (56)
#endif

#define MAXDV ((1ULL) << (MAXDEPTH))


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
        return d2 < other.d2; //MINIMIZING Norm 2, DIFFERENT from original
    }
};

struct BaseType{
    u64 Vec;
    u64 Dep;
    
     __host__ __device__
    bool operator<(const BaseType& other) const {
        return Vec < other.Vec;
    }

    __host__ __device__
    BaseType operator^(const BaseType& other) const {
        BaseType result;
        result.Vec = Vec ^ other.Vec;  
        result.Dep = Dep + other.Dep;  
        return result;
    }

    __host__ __device__
    BaseType operator+(const BaseType& other) const {
        BaseType result;
        result.Vec = Vec ^ other.Vec;
        result.Dep = ((Dep > other.Dep ? Dep : other.Dep) << 1);
        return result;
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
u64 *C; //C: Store the binomial coefficient at the gpu side and the cpu side. It will not be modified after initial calculation
u64 *Y; //Y: Store the targets both at the cpu and the gpu side
BaseType *B,*nextB; //B: current variables (bases) at cpu and gpu side. nextB: C(m,2) candidates at gpu side.
int *D,*ND;  //D: store the distances at cpu and gpu side. ND: store the values of the new distance vector after inserting each of the C(m,2) candidates into the current base (length: C(m,2)*n)
int prevD[MAXN];
//,deltaD=-1,deltaPrev=-1; //Store the previous distance, The change in distance and the maximum value of the index that caused the change. (for exEasyMove check)

// int ND[MAXN*MAXM*MAXM];  //for debug, no use.
int N,M; //matrix dimension and current variables.
BaseType *W[10]={nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr},**Wptr;
u64 *Wlen;
Metrics *E; //Store the struct of Metrics at GPU and CPU side.
char *R,*S; //Stores the results of binary search (initialized to 0, length: C(m,2); set to 1 if the search is successful) R for updating distance, S for check the effective of the candidates. they have the same function.
unsigned *OK; //Store whether each candidate meets the algorithm requirements (exEasymove in BP, and the minimum non-zero value in the distance vector in A1A2).
u64 globalAvailPairs; //number of pairs that meet the requirement


void initDealing()
{
    //Initialize variables at the beginning of each call of dealBP10()
	M=N;
	for(int i=0;i<N;i++) 
    {
        B[i].Vec=((1ULL)<<i);
        B[i].Dep=1;
    }
	for(int i=0;i<N;i++) D[i]=__builtin_popcountll(Y[i])-1;

	lenProg=0;
}
__global__ void getCombArrKernel2(BaseType* result,BaseType* base,const u64* combTable,int n,int m,u64 totalComb,int tableCols=10)
{
    unsigned long long k=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=totalComb)return;
    int s=0;
    int curM=m;
    unsigned long long curK=k;
    BaseType combVec={0,0};//calculate the k-th comb-seq value of C(n,m).
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
                combVec=combVec^base[x];
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
	Wlen[combNumber]=C[baseNumber*10+combNumber];
	if(W[combNumber]==nullptr) cudaMallocManaged(&W[combNumber],sizeof(BaseType)*C[baseNumber*10+combNumber]);
    int blockSize=256; //can be 512
    unsigned long long gridSize=(C[baseNumber*10+combNumber]+blockSize-1) / blockSize;
    getCombArrKernel2<<<gridSize, blockSize>>>(W[combNumber],B,C,baseNumber,combNumber,C[baseNumber*10+combNumber]);
	thrust::sort(thrust::device,W[combNumber],W[combNumber]+Wlen[combNumber]); //sort here.
}
void newDistancePre()
{
    //Update the W array in each iteration
	for(int i=0;i<10;i++)
	{
		if(W[i]!=nullptr)
		{
			cudaFree(W[i]);
			W[i]=nullptr;
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
		Wlen[i]=C[10*M+i];
		cudaMallocManaged(&W[i],sizeof(BaseType)*Wlen[i]);
		combEntry2(M,i);
	}
	cudaMemcpy(D,D,sizeof(int)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(Wlen,Wlen,sizeof(Wlen),cudaMemcpyHostToDevice);
	cudaMemcpy(Wptr,W,10ULL*sizeof(BaseType*),cudaMemcpyHostToDevice);
}

__global__ void getCandidateKernel(BaseType* CandidateBx,const BaseType* Bx,int mx,long long totalPairs,Metrics *Ex)
{
    //Parallel compute all C(m,2) candidates at gpu side.
    long long tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<totalPairs)
    {
        double m=mx;
        int i=(int)((2.0*m-1.0-sqrt((2.0*m-1.0)*(2.0*m-1.0)-8.0*tid))/2.0);
        long long rowStartIdx=1LL*i*mx-1LL*i*(i+1)/2;
        int j=(int)(tid-rowStartIdx+i+1);
        CandidateBx[tid]=Bx[i]+Bx[j];
        Ex[tid].a=i; //store the i and j value first
        Ex[tid].b=j;
        Ex[tid].d1=99999999;
        Ex[tid].d2=99999999;
    }
}
void getCandidateEntry()
{
    //Entry of getCandidateKernel
    u64 totalPairs=1LL*M*(M-1)/2;
    if (totalPairs<=0) return;
    int blockSize=256;
    long long gridSize=(totalPairs+blockSize-1)/blockSize;
    getCandidateKernel<<<gridSize,blockSize>>>(nextB,B,M,totalPairs,E);
}


__device__ char deviceBinSearch(const BaseType* arr,u64 len,u64 target,u64 costlim)
{
    //binary search at gpu side.
    if(len==0) return 0;
    long long L=0,R=(long long)len-1;
    while(L<=R)
    {
        long long M=L+(R-L)/2;
        if(arr[M].Vec==target)
            return arr[M].Dep<=costlim;
        if(arr[M].Vec<target) L=M+1;
        else R=M-1;
    }
    return 0;
}
__global__ void updDistance4Dis2(char *resultArray,
    const u64 Yxgpu,
    const BaseType *CandidateB,
    const BaseType *sortedBase,
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
    Tx=(Yxgpu^CandidateB[pairID].Vec^sortedBase[combID].Dep);
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
    const u64 Yxgpu,
    const BaseType *CandidateB,
    const BaseType *w_k1,
    const u64 totalComb_1,
    const BaseType *w_k2,   
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
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    BaseType tmp=CandidateB[pairID]^w_k2[combID];

    u64 Tx=Yxgpu^CandidateB[pairID].Vec^w_k2[combID].Vec;
    if(tmp.Dep>MAXDV) return;

    char r=deviceBinSearch(w_k1,totalComb_1,Tx,MAXDV-tmp.Dep); //binarysearch at GPU.
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



// --- MODIFIED KERNEL ---
// Added support for tagOKarr == nullptr (when not filtering)
__global__ void updDistance4Dis2A1(
        char *resultArray,
        const u64 Yxgpu,
        const BaseType *Bases,
        const BaseType *sortedBase,
        const u64 totalTasks,
        const u64 totalComb,
        const u64 totalPairs,
        unsigned *tagOKarr=nullptr, //filtered candidates
        const int m=0 //no use.
    )
{
    u64 tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=totalTasks)return;
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    int u,v;
    int pairInd=tagOKarr[pairID];
    u=pairInd>>16;
    v=pairInd&0xffff;
    BaseType tmp;
    tmp=(Bases[v]+Bases[u])^sortedBase[combID];
    if(tmp.Vec==Yxgpu&&tmp.Dep<=MAXDV)
    {
        resultArray[pairID]=1;
    }
}


// --- MODIFIED KERNEL ---
// Added support for tagOKarr == nullptr (when not filtering)
__global__ void updDistance4Dis3PlusA1(char *resultArray,
    const u64 Yxgpu,
    const BaseType *Bases,
    const BaseType *w_k1,
    const u64 totalComb_1,
    const BaseType *w_k2,   
    const u64 totalComb_2, 
    const u64 totalTasks,
    const u64 totalPairs,
    unsigned *tagOKarr, //filtered candidates
    const int m=0 // no use
    )
{

    u64 tid=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=totalTasks) return;
    u64 pairID=tid%totalPairs;
    u64 combID=tid/totalPairs;
    
    int u,v;
    
    int pairInd=tagOKarr[pairID];
    u=pairInd>>16;
    v=pairInd&0xffff;
    BaseType tmp=(Bases[v]+Bases[u])^w_k2[combID];
    u64 Tx=Yxgpu^tmp.Vec;
    u64 Cost=tmp.Dep;
    if(Cost>MAXDV) return;
    
    char r=deviceBinSearch(w_k1,totalComb_1,Tx,MAXDV-Cost);
    if(r)
        resultArray[pairID]=1;
}

void calculateDis4ezMoveEntry(u64 newBase, int a)
{
    //Deal with distance update in the scenario of easymove
    char R_tmp[MAXN];
    cudaMemset(R,0,N);
    D[a]=0;
    for(int i=0;i<=N;i++)
    {
        if(D[i]==0||D[i]==1) continue; //Note: Assume Y is full ranked, other D[i]==1 will note got to be 0.
        if(D[i]==2)
        {
            for(int j=0;j<M;j++)
            {
                if(!(Y[i]^newBase^B[j].Vec))
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
            cudaMemcpy(nextB,&newBase,sizeof(u64),cudaMemcpyHostToDevice);
            updDistance4Dis3Plus<<<gridSize,threadsPerBlock>>>(R+i,Y[i],nextB,W[K1],Wlen[K1],W[K2],Wlen[K2],totalTasks,1);
        }
    }
    cudaMemcpy(R_tmp,R,N,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
        D[i]=D[i]-R_tmp[i];
    cudaMemcpy(D,D,sizeof(int)*N,cudaMemcpyHostToDevice);
}

int easyMove()
{
    //Deal with easymove
	int a,found=0;
	int oneNumber=0,oneCount[64];
	for(a=0;a<N;a++)
		if(D[a]==1)
		{
			
			oneCount[found]=a;
			found++;
			//break;
		}
	if(!found) return 0;
	for(int iii=0;iii<found;iii++){
	a=oneCount[iii];
	for(int i=0;i<M;i++)
	{
        //if(B[i].Dep>=(1<<MAXDEPTH)) continue;
        for(int j=1+i;j<M;j++)
		{

			if(Y[a]==(B[i]^B[j]).Vec&&(B[i].Dep<(1<<MAXDEPTH))&&(B[j].Dep<(1<<MAXDEPTH)))
			{
                BaseType newBase = B[i] + B[j];
                cudaMemcpy(D+a, D+a, sizeof(int), cudaMemcpyHostToDevice);
                calculateDis4ezMoveEntry(newBase.Vec,a);
				programSeq[lenProg].t=M;
				programSeq[lenProg].a=i;
				programSeq[lenProg].b=j;
				programSeq[lenProg].y=a;
				lenProg++;
				B[M]=B[i]+B[j];
				cudaMemcpy(&B[M],&B[M],sizeof(BaseType),cudaMemcpyHostToDevice);
				printf("%d %d %d %d\n",M,i,j,a);
                M++;
				return 1;
			}
		}
		
    	}}
    return 0;
}
__global__ void calculateND4Yi(int *NDgpu,char *resultArr,u64 totalPairs,int targetID,int curDis,int Nx)
{
    u64 pairID=0ULL+1ULL*blockIdx.x*blockDim.x+threadIdx.x;
    if(pairID>=totalPairs) return;
    if(curDis<=1) NDgpu[pairID*Nx+targetID]=curDis;
    else NDgpu[pairID*Nx+targetID]=curDis-resultArr[pairID];
}

// Added useFilter parameter
void computeAllDistanceEntry()
{
    u64 totalPairs = globalAvailPairs;
    // unsigned* okPtr = OK;//false ? OK : nullptr; // Determine which pointer to pass
    // BaseType *baseArr=B;//false ?B:nextB;

    for(int i=0;i<N;i++)
    {
        int Di=D[i];
        if(Di==2)
        {
            cudaMemset(R,0,totalPairs);
            u64 totalTasks=totalPairs*M;
            int threadsPerBlock = 256;
            u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
            updDistance4Dis2A1<<<gridSize,threadsPerBlock>>>(R,Y[i],B,B,totalTasks,M,totalPairs,OK,M);
        }
        if(Di>=3)
        {
            cudaMemset(R,0,totalPairs);
            int K=Di-1;
            int K2=K/2;
            int K1=K-K2;
            u64 totalTasks=totalPairs*Wlen[K2];
            int threadsPerBlock=256;
            u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
            updDistance4Dis3PlusA1<<<gridSize,threadsPerBlock>>>(R,Y[i],B,W[K1],Wlen[K1],W[K2],Wlen[K2],totalTasks,totalPairs,OK,M);
        }        
        u64 totalTasks=totalPairs*1;
        int threadsPerBlock=256;
        u64 gridSize=(totalTasks+threadsPerBlock-1)/threadsPerBlock;
        calculateND4Yi<<<gridSize,threadsPerBlock>>>(ND,R,totalTasks,i,Di,N);
    }
}

__global__ void checkCandiateKernel(
    const BaseType *CandidateB,
    const u64 totalPairs,
    unsigned *tagOKarr,
    const u64 m
)
{
    u64 pairID = 0ULL+1ULL*blockIdx.x * blockDim.x + threadIdx.x;
    if (pairID >= totalPairs) return;
    double mf = (double)m;
    int i = (int)((2.0 * mf - 1.0 - sqrt((2.0 * mf - 1.0) * (2.0 * mf - 1.0) - 8.0 * pairID)) / 2.0);
    long long rowStartIdx = 1LL * i * m - 1LL * i * (i + 1) / 2;
    int j = (int)(pairID - rowStartIdx + i + 1);
    if (CandidateB[pairID].Dep <= (MAXDV >> 1))
        tagOKarr[pairID] = ((i << 16) | (j & 0xffff));
    else
        tagOKarr[pairID] = 0xffffffff;
}

u64 checkCandidateEntry()
{
    u64 totalPairs=1LL*M*(M-1)/2;
    cudaMemset(OK,-1,sizeof(unsigned)*totalPairs);
    
    int blockSize=256;
    long long gridSize=(totalPairs+blockSize-1)/blockSize;
    checkCandiateKernel<<<gridSize, blockSize>>>(nextB,totalPairs,OK,M);
    thrust::sort(thrust::device,OK,OK+totalPairs);
    unsigned* bound_ptr = thrust::lower_bound(thrust::device, OK, OK + totalPairs, 0x70000000U);
    u64 count = bound_ptr - OK;

    return count;
}


__global__ void calcuteMetricsKernel(
    Metrics* E,           // 输出：指标数组
    const int* ND,        // 输入：所有距离结果
    int n,                  // 目标点数量 N
    int m,                  // 当前基的数量 M
    long long totalPairs,
    unsigned *OK=nullptr
    )   // 总组合数 M*(M-1)/2
{
    long long k = 0ULL+1ULL*blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= totalPairs) return;

    // 1. 还原 i 和 j (基于 M*(M-1)/2 的排布逻辑)
    // 采用数学解析法，避免在 Kernel 里跑循环
    int i,j;
    if(OK==nullptr)
    {
        double mf = (double)m;
        i = (int)((2.0 * mf - 1.0 - sqrt((2.0 * mf - 1.0) * (2.0 * mf - 1.0) - 8.0 * k)) / 2.0);
        long long rowStartIdx = 1LL * i * m - 1LL * i * (i + 1) / 2;
        j = (int)(k - rowStartIdx + i + 1);
    }
    else
    {
        unsigned tmp=OK[k];
        i=(tmp>>16);
        j=(tmp&0xffff);
    }
    // 2. 访问 ND 并计算 d1, d2
    int sumD = 0;
    int sumN = 0;
    
    // 定位到 ND 中属于该组合 k 的起始位置
    const int* current_NRow = &ND[k * n];

    for (int tid = 0; tid < n; tid++) {
        int val = current_NRow[tid];
        sumD += val;
        sumN += val * val;
    }

    // 3. 填充结构体
    E[k].a = i;
    E[k].b = j;
    E[k].ind=k;
    E[k].d1 = sumD;
    E[k].d2 = sumN;
}

// --- MODIFIED FUNCTION ---
// Added useFilter parameter
int calcuteMetricsEntry(bool useFilter)
{
    u64 totalPairs=globalAvailPairs;
    int blockSize=256;
    u64 gridSize=(totalPairs+blockSize-1)/blockSize;
    
    unsigned* okPtr=useFilter?OK:nullptr;

    calcuteMetricsKernel<<<(int)gridSize,blockSize>>>(E,ND,N,M,totalPairs,okPtr);
    // thrust::sort(thrust::device, E, E+totalPairs,MetricsComparator()); use thrust::sort is also ok, but it might not that stable and the data scale here is not that large.
    cudaMemcpy(E,E,sizeof(Metrics)*totalPairs,cudaMemcpyDeviceToHost);
    sort(E,E+totalPairs);
    int k=0;
    if(totalPairs>0)
    {
        Metrics target=E[0];
        
        long long L=0,R=totalPairs-1;
        long long ans=0;
        
        while(L<=R)
        {
            long long mid=L+(R-L)/2;
            if(E[mid].d1==target.d1&&E[mid].d2==target.d2)
            {
                ans=mid;
                L=mid+1;
            }
            else
                R=mid-1;
        }
        k=(int)ans+1;
    }
    return k;
}

char isUseful[MAXM];
int reNumber[MAXM],reNumberCount=0;
int finalDepth[MAXM];
void prunUseless()
{
    memset(isUseful,0,sizeof(isUseful));
    memset(reNumber,-1,sizeof(reNumber));
    for(int i=lenProg-1;i>=0;i--)
    {
        int ai=programSeq[i].a,bi=programSeq[i].b,yi=programSeq[i].y;
        if(isUseful[i])
        {
            if(ai>=N) isUseful[ai-N]=1;
            if(bi>=N) isUseful[bi-N]=1;
        }
        else if(~programSeq[i].y)
        {
            isUseful[i]=1;
            if(ai>=N) isUseful[ai-N]=1;
            if(bi>=N) isUseful[bi-N]=1;
        }
    }
    for(int i=0;i<N;i++) reNumber[i]=i;
    reNumberCount=N;
    for(int i=0;i<N;i++) finalDepth[i]=0;
    sprintf(OUTFILENAME,"Record%s-%d-%06x.ans",IFILENAME,REC,RUNID);
    FILE *fp=fopen(OUTFILENAME,"w");
    for(int i=0;i<lenProg;i++)
    {
        if(isUseful[i])
        {
            int ti,ai,bi,yi,di;
            ti=programSeq[i].t;
            ai=programSeq[i].a;
            bi=programSeq[i].b;
            yi=programSeq[i].y;
            reNumber[ti]=reNumberCount++;
            finalDepth[reNumber[ti]]=1+(
                (finalDepth[reNumber[ai]]>finalDepth[reNumber[bi]])?
                finalDepth[reNumber[ai]]:
                finalDepth[reNumber[bi]]
            );
            di=finalDepth[reNumber[ti]];

            fprintf(fp,"%d %d %d %d %d\n",reNumber[ti],reNumber[ai],reNumber[bi],yi,di);
        }
    }
    fclose(fp);
}
void dealBP10()
{
    sprintf(OUTFILENAME,"Record%s-%d-%06x.bns",IFILENAME,++REC,RUNID);
	sprintf(LOGFILENAME,"Record%s-%d-%06x.log",IFILENAME,REC,RUNID);
	unsigned long long TotalTimeCost=0;
	FILE *fp,*flp;
    auto T1=chrono::high_resolution_clock::now(),T2=chrono::high_resolution_clock::now();
	initDealing();

	for(int iteration=0;M<MAXM-N-5;iteration++)
	{
        printf("[%d]\n",iteration);
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
        computeAllDistanceEntry();
        
		int bestCount=calcuteMetricsEntry(1);
        
        Metrics selected=E[rand()%bestCount];
        int besti=selected.a,bestj=selected.b;
        int bestOffset=selected.ind;

        memcpy(prevD,D,sizeof(int)*N);
        cudaMemcpy(D,ND+(N*bestOffset),sizeof(int)*N,cudaMemcpyDeviceToHost);
        programSeq[lenProg].t=M;
        programSeq[lenProg].a=besti;
        programSeq[lenProg].b=bestj;
        programSeq[lenProg].y=-1; 
        lenProg++;
        B[M]=B[besti]+B[bestj];
        cudaMemcpy(B+M,B+M,sizeof(BaseType),cudaMemcpyHostToDevice); 
        fprintf(fp,"%d %d %d %d\n",M,besti,bestj,-1);
        fclose(fp);
        M++;
	}
    if(M<MAXM-N-5)
        prunUseless();
}
void initGlobals()
{
	cudaMallocManaged(&C,10ULL*sizeof(u64)*MAXM);
	C[0]=1;
    for(int i=0;i<MAXM;i++)
        for(int j=0;j<=i&&j<10;j++)
        {
            if(j==0||j==i) C[i*10+j]=1;
            else
                C[i*10+j]=C[(i-1)*10+(j-1)]+C[(i-1)*10+j];
        }
	cudaMallocManaged(&B,sizeof(BaseType)*MAXM);
	cudaMallocManaged(&nextB,sizeof(BaseType)*MAXM*MAXM);
	cudaMallocManaged(&Y,sizeof(u64)*MAXN);
	cudaMallocManaged(&D,sizeof(int)*MAXN);
	cudaMallocManaged(&ND,sizeof(int)*MAXN*MAXM*MAXM);
	cudaMallocManaged(&Wlen,sizeof(u64)*10);
	cudaMallocManaged(&Wptr,sizeof(BaseType*)*10);
	cudaMallocManaged(&E,sizeof(Metrics)*MAXM*MAXM);
    cudaMallocManaged(&R,sizeof(char)*MAXM*MAXM);
    cudaMallocManaged(&S,sizeof(char)*MAXM*MAXM);
    cudaMallocManaged(&OK,sizeof(unsigned)*MAXM*MAXM);
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
    initGlobals();
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
	
    unsigned short runtime=RUNTIME;
	while(runtime--)
    {
        dealBP10();
        for(int i=0;i<M;i++)
        {
            printf("%016X %016X\n",B[i].Vec,B[i].Dep);
        }
    }
}
