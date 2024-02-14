#include "fn2v.h"


void node2vec(PWNet& InNet, double** TransMtrx,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  TVVec<TInt, int64>& WalksVV,
  TIntFltVH& EmbeddingsHV) {
  //Preprocess transition probabilities
  PreprocessResTransitionProbs(InNet, TransMtrx, Verbose);
  TIntV NIdsV;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    NIdsV.Add(NI.GetId());
  }
  //Generate random walks
  int64 AllWalks = (int64)NumWalks * NIdsV.Len();
  WalksVV = TVVec<TInt, int64>(AllWalks,WalkLen);
  TRnd Rnd(time(NULL));
  int64 WalksDone = 0;
  for (int64 i = 0; i < NumWalks; i++) {
    NIdsV.Shuffle(Rnd);
#pragma omp parallel for schedule(dynamic)
    for (int64 j = 0; j < NIdsV.Len(); j++) {
      if ( Verbose && WalksDone%10000 == 0 ) {
        printf("\rWalking Progress: %.2lf%%",(double)WalksDone*100/(double)AllWalks);fflush(stdout);
      }
      TIntV WalkV;
      SimulateResidualWalk(InNet, NIdsV[j], WalkLen, Rnd, WalkV);
      for (int64 k = 0; k < WalkV.Len(); k++) { 
        WalksVV.PutXY(i*NIdsV.Len()+j, k, WalkV[k]);
      }
      WalksDone++;
    }
  }
  if (Verbose) {
    printf("\n");
    fflush(stdout);
  }
  TVVec<TInt, int64> ModelWalksVV = WalksVV;
  //Learning embeddings
  LearnEmbeddings(ModelWalksVV, Dimensions, WinSize, Iter, Verbose, EmbeddingsHV);
}


double **cr_matrix(int n, int m)
{
	double **matrix = new double*[n];
	for(int i=0; i<n; i++)
	{
		matrix[i] = new double[m];
	}
	return matrix;
}

void calc_initial_transmtrx(graph g, double ** initial_transmtrx, double phi)
{
	int nnodes = g.get_num_nodes();
	node_t *nodes = g.get_nodes();
	for(int i=0; i<nnodes; i++)
	{
		for(int j=0; j<nnodes; j++)
		{
			if(nodes[i].out_neighbor_ids.find(j) != nodes[i].out_neighbor_ids.end())
			{
				if(nodes[i].residual_type == protected_attr && g.count_out_neighbors_with_community(i, !protected_attr) != 0)
					initial_transmtrx[i][j] = (1 - phi) / g.count_out_neighbors_with_community(i, !protected_attr);
				else if(nodes[i].residual_type == !protected_attr && g.count_out_neighbors_with_community(i, protected_attr) != 0)
					initial_transmtrx[i][j] = phi / g.count_out_neighbors_with_community(i, protected_attr);
			}
			else
				initial_transmtrx[i][j] = 0;
		}
	}
	return;
}

void calc_transmtrx(graph g, double **transmtrx, double **initial_transmtrx, double *residuals, std::vector<double> probs)
{
	int nnodes = g.get_num_nodes();
	node_t *nodes = g.get_nodes();
	for(int i=0; i<nnodes; i++)
	{
		for(int j=0; j<nnodes; j++)
		{
			transmtrx[i][j] = initial_transmtrx[i][j] + (residuals[i] * probs[j]);
		}
	}
	return;
}


void set_residual_types(graph g, double phi)
{
	int nnodes = g.get_num_nodes();
	node_t *nodes = g.get_nodes();
	for(int i=0; i<nnodes; i++)
	{
		if(g.count_out_neighbors_with_community(i, 1)/g.get_out_degree(i) < phi)
			nodes[i].residual_type = 1;
		else
			nodes[i].residual_type = 0;
	}
	return;
}

double *calc_residuals(graph g, double phi)
{
	int nnodes = g.get_num_nodes();
	node_t *nodes = g.get_nodes();
	double *residuals = new double[nnodes];
	for(int i=0; i<nnodes; i++)
	{
		if(nodes[i].residual_type == 1)
			if(g.count_out_neighbors_with_community(i, 0) != 0)
				residuals[i] = phi - (((1 - phi) * g.count_out_neighbors_with_community(i, 1)) / (g.count_out_neighbors_with_community(i, 0)));
			else
				residuals[i] = phi;
		else
			if(g.count_out_neighbors_with_community(i, 1) != 0)
				residuals[i] = (1 - phi) - ((phi * g.count_out_neighbors_with_community(i, 0)) / g.count_out_neighbors_with_community(i, 1));
			else
				residuals[i] = 1 - phi;
	}

	return residuals;
}


void PreprocessNode (PWNet& InNet, double* TransMtrx, TWNet::TNodeI NI, int64& NCnt, const bool& Verbose) {
  if (Verbose && NCnt%100 == 0) {
    printf("\rPreprocessing progress: %.2lf%% ",(double)NCnt*100/(double)(InNet->GetNodes()));fflush(stdout);
  }
  TFltV PTable;                              //Probability distribution table
  for(int j=0; j<InNet->GetNodes(); j++) {
    PTable.Add(TransMtrx[j]);
  }
  GetNodeAlias(PTable,NI.GetDat().GetDat(NI.GetId()));
  
  NCnt++;
}


void PreprocessResTransitionProbs(PWNet& InNet, double** TransMtrx, const bool& Verbose) {
  TIntV NIds;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    InNet->SetNDat(NI.GetId(),TIntIntVFltVPrH());
    NI.GetDat().AddDat(NI.GetId(), TPair<TIntV, TFltV>(TIntV(InNet->GetNodes()), TFltV(InNet->GetNodes())));
    NIds.Add(NI.GetId());
  }
 
  int64 NCnt = 0;
#pragma omp parallel for schedule(dynamic)
  for (int64 i = 0; i < NIds.Len(); i++) {
    PreprocessNode(InNet, TransMtrx[i], InNet->GetNI(NIds[i]), NCnt, Verbose);
  }
  if(Verbose){ printf("\n"); }
}

int64 PredictResidualMemReq(PWNet& InNet) {
  int64 MemNeeded = 0;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    MemNeeded += InNet->GetNodes()*(sizeof(TInt) + sizeof(TFlt));
  }
  return MemNeeded;
}

void SimulateResidualWalk(PWNet& InNet, int64 StartNId, const int& WalkLen, TRnd& Rnd, TIntV& WalkV) {
  WalkV.Add(StartNId);
  if (WalkLen == 1) { return; }
  while (WalkV.Len() < WalkLen) {
    int64 Next = AliasDrawInt(InNet->GetNDat(WalkV.Last()).GetDat(WalkV.Last()), Rnd);
    WalkV.Add(Next);
  }
}
