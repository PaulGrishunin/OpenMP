/*
 * Experiment.cpp
 *
 *  Created on: 16 maj 2016
 *      Author: oramus
 */

#include<stdlib.h>
#include<iostream>
#include "Experiment.h"
#include "Distribution.h"
#include <omp.h>
#include <time.h>
#define DEBUG_ON_

using namespace std;

struct drand48_data drand_buf;

#pragma omp threadprivate(drand_buf)


Experiment::Experiment(int balls, int drawsNumber) {
	this->balls = balls;
	this->drawsNumber = drawsNumber;
    
    #pragma omp parallel
    {
    int seed = omp_get_wtime()*10000 + omp_get_thread_num() * 1999;
    srand48_r(seed, &drand_buf);
    }
    
	hmax = 0; // wyznaczamy maksymalna sume
	hmin = 0; // i najmniejsza sume
	for (int i = 0; i < drawsNumber; i++) {
		hmax += balls - i;
		hmin += i + 1; // 1 + 2 + 3 + ... liczba losowan
	}

	cout << "Histogram min: " << hmin << " max: " << hmax << endl;

	histogram = new long[hmax + 1];

	for (long i = 0; i < hmax + 1; i++)
		histogram[i] = 0;
}

void Experiment::clearUsed() {
//	for (int i = 0; i < balls; i++)
//		used[i] = false;
}

long Experiment::singleExperimentResult() {
	long sum = 0;
	int ball;
	double p;

    bool *used = new bool[balls];
    for (int i = 0; i < balls; i++)
        used[i] = false;

    

    for (int i = 0; i < drawsNumber;) {
        double x;
        drand48_r(&drand_buf, &x);
        ball = 1 + (int)(balls * x); // rand losuje od 0 do RAND_MAX wlacznie
        if (used[ball - 1])
            continue;
        p = Distribution::getProbability(i + 1, ball); // pobieramy prawdopodobienstwo wylosowania tej kuli
        drand48_r(&drand_buf, &x);
        if (x < p) // akceptacja wyboru kuli z zadanym prawdopodobienstwem
        {
            #ifdef DEBUG_ON
                cout << "Dodano kule o numerze " << ball << endl;
            #endif
            used[ball - 1] = true;
            sum += ball; // kule maja numery od 1 do balls wlacznie
            ++i;
        }
    }

///	cout << "Suma = " << sum << endl;
    delete[] used;
	return sum;
}

Result * Experiment::calc(long experiments) {
    
     
    #pragma omp parallel   //private(balls)
        {
        
            #pragma omp for nowait
        for (long l = 0; l < experiments; l++) {
            long sum = singleExperimentResult();
            #pragma omp atomic
            histogram[sum]++;
        }
        
    
        }
	long maxID = 0;
	long minID = 0;
	long maxN = 0;
	long minN = experiments;
	double sum = 0.0;
	long values = 0;

    double sum_local = 0.0;
    long maxN_local = 0;
    long maxID_local = 0;
    long values_local = 0;    
  
    
    #pragma omp parallel firstprivate( maxN_local, maxID_local, sum_local, values_local )
    {
    #pragma omp for nowait
	for (long idx = hmin; idx <= hmax; idx++) {
		if (maxN_local < histogram[idx]) {
			maxN_local = histogram[idx];
			maxID_local = idx;
		}
		sum_local += idx * histogram[idx];      
		values_local += histogram[idx];
    }
    #pragma omp critical
        {
        if ( maxN < maxN_local ) maxN = maxN_local;
        if ( maxID < maxID_local ) maxID = maxID_local;
		//if ( sum < sum_local ) sum = sum_local;
       // if ( values < values_local ) values = values_local;
        
        sum = sum + sum_local;
        values = values + values_local;
        
//     #pragma omp atomic
//         sum += sum_local;
// 		values += values_local;
        
        }
    
    }
        
// indeks to wartosc, histogram -> liczba wystapien
	return new Result(maxID, maxN, sum / values, values);

}
Experiment::~Experiment() {
	delete[] histogram;
//	delete[] used;
}
