#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <cassert>
#include <map>
#include <set>
#include <chrono>
#define u64 unsigned 
using namespace std;

#define MAXN 36
#define MAXM 3333
u64 B0[MAXM],Y[MAXN];
u64 B[MAXM],*Bgpu;

int n,m,m0,lenProg;
char OUTFILENAME[160];
int D[MAXN],oldDist[MAXN],oneCount,oneDis[MAXN];
int ND[MAXN];
struct Program{int t,a,b,y;}programSeq[MAXM];
vector<int> bestX,bestY,bestND[MAXN];
// int bestCount=0,maxBestCount=0;
int BEST=99999999;

unsigned char buffer[8192]; 
//this is a 8KB space to avoid buffer overflow, only for debug and has no other use.
//the context of the buffer should be always zeros.
unsigned long long C[MAXM*9],*Cgpu;
map<u64,int>mpp;
int REC=0;
unsigned long long RUNID;

int bitCount(u64 x)
{
	int countbit=0;
	while(x)
	{
		x-=(x&(-x));
		countbit++;
	}
	return countbit;
}
void initGlobals()
{
	mpp.clear();
	for(int i=0;i<n;i++) B[i]=(1<<i);
	cudaMemcpy(Bgpu,B,4*n,cudaMemcpyHostToDevice);
	for(int i=0;i<n;i++) mpp[Y[i]]=1+i;
    for(int i=0;i<n;i++) D[i]=bitCount(Y[i])-1;
	m=n;
	lenProg=0;
}


__global__ void getCombArrKernel(int* result, const unsigned long long* comb_table, int n, int m, unsigned long long total_comb, int table_cols=9) {
    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total_comb) return;

    int s = 0; 
    int cur_m = m; 
    unsigned long long cur_k = k; 


    for (int idx = 0; idx < m; ++idx) {
        int x = s;
        while (true) {
            int remaining = n - x - 1;
            unsigned long long num_comb = 0;

            if (cur_m - 1 <= 0) {
                num_comb = 1;
            } else if (remaining >= (cur_m - 1)) {
                num_comb = comb_table[remaining * table_cols + (cur_m - 1)];
            } else {
                num_comb = 0;
            }

            if (cur_k < num_comb) {
                result[k * m + idx] = x;
                s = x + 1;
                cur_m--;
                break;
            } else {
                cur_k -= num_comb;
                x++;
            }
        }
    }
}

__global__ void getXorSumKernel(unsigned int* xor_results, const unsigned int* B, 
                                    const int* combinations, unsigned long long total_comb, int K) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_comb) return;

    unsigned int xor_sum = 0;

    for (int i = 0; i < K; i++) {
        int element_index = combinations[idx * K + i];
        xor_sum ^= B[element_index];
    }
    xor_results[idx] = xor_sum;
}


void combEntry(int baseNumber,int combNumber,u64* retArray)
{
    int *CDatagpu=nullptr;
    cudaMalloc(&CDatagpu,4LL*combNumber*C[baseNumber*9+combNumber]);
    int blockSize = 256;
    unsigned long long gridSize = (C[baseNumber*9+combNumber] + blockSize - 1) / blockSize;
    getCombArrKernel<<<gridSize, blockSize>>>(CDatagpu, Cgpu, baseNumber, combNumber, C[baseNumber*9+combNumber]);
    unsigned *BResultgpu=nullptr;
    cudaMalloc(&BResultgpu,4*C[baseNumber*9+combNumber]);
    getXorSumKernel<<<gridSize,blockSize>>>(BResultgpu,Bgpu,CDatagpu,C[baseNumber*9+combNumber],combNumber);
    cudaMemcpy(retArray,BResultgpu,4*C[baseNumber*9+combNumber],cudaMemcpyDeviceToHost);
    cudaFree(CDatagpu);
    cudaFree(BResultgpu);
}
u64 *W[9]={nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
set<u64>tpp[9];
void newDistancePre()
{
	for(int i=0;i<9;i++)
		if(W[i]!=nullptr)
		{
			delete (W[i]);
			W[i]=nullptr;
		}
	for(int i=0;i<9;i++)
		tpp[i].clear();
	int dmax=0;
	for(int i=0;i<n;i++)
		if(D[i]>dmax) dmax=D[i];
	int total=(dmax)/2;
	printf("DMAX=%d,TOTAL=%d\n",dmax,total);

	for(int i=1;i<=total;i++)
	{
		W[i]=new u64[C[9*m+i]];
		combEntry(m,i,W[i]);
	}
	for(int i=1;i<=total;i++)
	{
		tpp[i].clear();
		for(int j=0;j<C[9*m+i];j++)
			tpp[i].insert(W[i][j]);
	}
}
bool reachableFaster(u64 T,int K)
{
	if(K==0) return 0;
	if(K==1)
	{
		for(int i=0;i<m;i++)
			if(B[i]==T) return 1;
		return 0;
	}
	int K1=K/2;
	int K2=K-K1;
	for(auto ip=tpp[K2].begin();ip!=tpp[K2].end();ip++)
	{
		auto tppk2i=*ip;
		auto pos=tpp[K1].find(T^tppk2i);
		if(pos!=tpp[K1].end()) return 1;
	}
	return 0;
}

int newDistance(int x,u64 newBase)
{
	if(D[x]==0) return 0;
	for(int i=0;i<m;i++)
		if(Y[x]==B[i]) return 0;
	if (reachableFaster(Y[x] ^ newBase, D[x] - 1))
        return D[x] - 1;
    else
        return D[x];
}

#ifndef TIMUID
#define TIMUID 0
#endif
void inputPreSteps()
{
	int v1,v2,v3,v4;
	while(1)
	{
		scanf("%d%d%d%d",&v1,&v2,&v3,&v4);
		if(v1+v2+v3+v4<0) break;
		programSeq[lenProg].t=v1;
		programSeq[lenProg].a=v2;
		programSeq[lenProg].b=v3;
		programSeq[lenProg].y=v4;
		++lenProg;
		B[m]=B[v3]^B[v2];
		cudaMemcpy(&Bgpu[m],&B[m],4,cudaMemcpyHostToDevice);
		++m;
	}
	for(int i=0;i<n;i++) scanf("%d",&D[i]);
}
void dealBP10()
{

	sprintf(OUTFILENAME,"T%d-%d-%llx.txt",TIMUID,++REC,RUNID);
	FILE *fp;
	initGlobals();
	// inputPreSteps();
	auto T1 = chrono::high_resolution_clock::now();
	for(;;)
	{
		auto T2=chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<std::chrono::milliseconds>(T2 - T1);
		printf("Last Time Cost: %08llu ms\n",duration.count());
		T1=T2;
		newDistancePre();
		printf("culcated dis.");
		int detect=0;
		for(int i=0;i<8192;i++)
			detect+=(buffer[i]!=0);
		printf("DETECT=%d\n",detect);
		if(detect)
		{
			for(int i=0;i<8192;i++)
				printf("%02x\n",buffer[i]&0xff);
		}
		fp=fopen(OUTFILENAME,"a");
		int sumDist=0;
		for(int i=0;i<n;i++) sumDist+=D[i];
		printf("%d ",sumDist);
		for(int i=0;i<n;i++)
			printf("%d%c",D[i],(i+1==n)?10:32);
		if(!sumDist) break;
		int found=0;
		for(int i=0;i<n;i++)
		{
			if(D[i]-1) continue;
			int a=i;
			for(int j=0;j<m;j++)
			{
				for(int k=1+j;k<m;k++)
				{
					if(Y[a]==(B[j]^B[k]))
					{
						programSeq[lenProg].t=m;
						programSeq[lenProg].a=j;
						programSeq[lenProg].b=k;
						programSeq[lenProg].y=a;
						++lenProg;
						B[m]=B[j]^B[k];
						found=1;
						cudaMemcpy(&Bgpu[m],&B[m],4,cudaMemcpyHostToDevice);
						fprintf(fp,"%d %d %d %d\n",m++,j,k,a);
						fclose(fp);
						break;
					}
				}
				if(found)
					break;
			}
			if(found)
				break;
		}
		if(found)
		{
			printf("Connect 0");
			for(int i=0;i<n;i++)
				D[i]=newDistance(i,B[m-1]);
			printf("Connect 1");
			continue;
		}
		int bestDist=99999999,bestNorm=-1;

		int vecsize=30000;
		if(m*m>vecsize) vecsize=m*m;
		bestX.clear();
		bestX.reserve(vecsize);
		bestY.clear();
		bestY.reserve(vecsize);
		
		for(int i=0;i<m;i++)
			for(int j=1+i;j<m;j++)
			{
				u64 newBase=B[i]^B[j];
				for(int k=0;k<n;k++)
					ND[k]=newDistance(k,newBase);
				int thisDist=0,thisNorm=0;
				for(int k=0;k<n;k++)
				{
					thisDist+=ND[k];
					thisNorm+=ND[k]*ND[k];
				}
				if(thisDist<bestDist||thisDist==bestDist&&thisNorm>bestNorm)
				{
					bestX.clear();
					bestY.clear();
					bestX.push_back(i);
					bestY.push_back(j);
					bestDist=thisDist;
					bestNorm=thisNorm;
					for(int k=0;k<n;k++)
					{
						bestND[k].clear();
						bestND[k].push_back(ND[k]);
					}
				}
				else if(thisDist==bestDist&&thisNorm==bestNorm)
				{
					bestX.push_back(i);
					bestY.push_back(j);
					for(int k=0;k<n;k++)
						bestND[k].push_back(ND[k]);
				}
			}
		int bestElemID=rand()%(bestND[0].size());
		int bi=bestX[bestElemID],bj=bestY[bestElemID];
		u64 tpc=B[bi]^B[bj];
		B[m]=tpc;
		for(int i=0;i<n;i++)
			D[i]=bestND[i][bestElemID];
		programSeq[lenProg++]={m,bi,bj,mpp[tpc]-1};
		cudaMemcpy(&Bgpu[m],&B[m],4,cudaMemcpyHostToDevice);
		fprintf(fp,"%d %d %d %d\n",m,bi,bj,mpp[tpc]-1);
		fclose(fp);
		m++;
	}
}
int main()
{
	srand(time(0));
	for(int i=0;i<6;i++)
	{
		RUNID<<=15;
		RUNID|=(rand()&0x7fff);
	}
	char *IFILENAME=(char*)malloc(33);
	sprintf(IFILENAME,"data%d.txt",TIMUID);
	FILE *fp=fopen(IFILENAME,"r");

	C[0]=1;
    for(int i=0;i<180;i++)
        for(int j=0;j<=i&&j<9;j++)
        {
            if(j==0||j==i) C[i*9+j]=1;
            else
                C[i*9+j]=C[(i-1)*9+(j-1)]+C[(i-1)*9+j];
        }
	cudaMalloc(&Cgpu,180*72);
    cudaMemcpy(Cgpu,C,180*72,cudaMemcpyHostToDevice);
	cudaMalloc(&Bgpu,MAXM*4);
	printf("%p %p\n",Cgpu,Bgpu);
	fscanf(fp,"%d",&n);
	m=n;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			int x;
			fscanf(fp,"%d",&x);
			if(x) Y[i]|=(1<<j);
		}
		mpp[Y[i]]=i+1;
	}

	fclose(fp);
	printf("end of input\n");
	m0=m;
	u64 T=0;
	while(--T) dealBP10();
	return 0;
}

