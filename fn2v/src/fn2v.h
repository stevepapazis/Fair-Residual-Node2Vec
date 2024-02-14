#ifndef FN2V_H
#define FN2V_H

#include "stdafx.h"
#include "Snap.h"
#include "n2v.h"
#include "biasedrandomwalk.h"
#include "word2vec.h"
#include "graph.hpp"
#include <vector>


typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

extern int protected_attr;


void node2vec(PWNet& InNet, double** TransMtrx, const int& Dimensions, const int& WalkLen, const int& NumWalks, const int& WinSize, const int& Iter, const bool& Verbose, TVVec<TInt, int64>& WalksVV, TIntFltVH& EmbeddingsHV);
double **cr_matrix(int n, int m);
void calc_initial_transmtrx(graph g, double ** initial_transmtrx, double phi);
void calc_transmtrx(graph g, double **transmtrx, double **initial_transmtrx, double *residuals, std::vector<double> probs);
void set_residual_types(graph g, double phi);
double *calc_residuals(graph g, double phi);
void PreprocessNode (PWNet& InNet, double* TransMtrx, TWNet::TNodeI NI, int64& NCnt, const bool& Verbose);
void PreprocessResTransitionProbs(PWNet& InNet, double** TransMtrx, const bool& verbose);
void SimulateResidualWalk(PWNet& InNet, int64 StartNId, const int& WalkLen, TRnd& Rnd, TIntV& Walk);
int64 PredictResidualMemReq(PWNet& InNet);


#endif
