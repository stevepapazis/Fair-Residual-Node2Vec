#include "stdafx.h"
#include "n2v.h"
#include "fn2v.h"
#include "graph.hpp"
#include "stochastic_random_search.h"
#include <fstream>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <set>
#include <math.h>


int MAX_ITERATIONS = 3;
int NUM_POINTS = 3;
int NUMBER_OF_DIRECTIONS = 2;
double PRECISION_OF_SOLUTION = pow(10, -8);
double PRECISION_OF_CAT_RATIO = pow(10, -4);
int TIMES_TO_TRY_FOR_INITIAL_POINT = 100;
double INITIAL_STEP = 1.0;
int SMALLER_ALLOWED_STEP = pow(10, -10);

int protected_attr;


struct n2v_args
{
	PWNet net;
	double **initial_transmtrx;
	double **transmtrx;
	double *residuals;
	TIntFltVH embeddings;
	TVVec <TInt, int64> walks;
	int dim;
	int walklen;
	int numwalks;
	int winsize;
	int iter;
	bool directed;
	bool weighted;
	bool verbose;
};


void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& InFileAttr, TStr& EmbsOutFile, TStr& WalksOutFile, int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose, double& ParamP, double& ParamQ, int& PrAttr, double& phi, bool& Directed, bool& Weighted)
{
	Env = TEnv(argc, argv, TNotify::StdNotify);
	Env.PrepArgs(TStr::Fmt("\nAn algorithmic framework for representational learning on graphs."));
	InFile = Env.GetIfArgPrefixStr("-i:", "graphs/fb_edges.txt", "Input graph path");
	InFileAttr = Env.GetIfArgPrefixStr("-iattr:", "graphs/fb_genders.txt", "Input attributes path");
	EmbsOutFile = Env.GetIfArgPrefixStr("-oe:", "graphs/fb_fn2v_embs.txt", "Output graph embs path");
	WalksOutFile = Env.GetIfArgPrefixStr("-ow:", "graphs/fb_fn2v_walks.txt", "Output graph walks path");
	Dimensions = Env.GetIfArgPrefixInt("-d:", 128, "Number of dimensions. Default is 128");
	WalkLen = Env.GetIfArgPrefixInt("-l:", 80, "Length of walk per source. Default is 80");
	NumWalks = Env.GetIfArgPrefixInt("-r:", 10, "Number of walks per source. Default is 10");
	WinSize = Env.GetIfArgPrefixInt("-k:", 10, "Context size for optimization. Default is 10");
	Iter = Env.GetIfArgPrefixInt("-e:", 1, "Number of epochs in SGD. Default is 1");
	PrAttr = Env.GetIfArgPrefixInt("-pattr:", 0, "Number of epochs in SGD. Default is 0");
	phi = Env.GetIfArgPrefixFlt("-phi:", 0.5, "Phi parameter. Default is 0.5");
	ParamP = Env.GetIfArgPrefixFlt("-p:", 1, "Return hyperparameter. Default is 1");
	ParamQ = Env.GetIfArgPrefixFlt("-q:", 1, "Inout hyperparameter. Default is 1");
	Verbose = Env.IsArgStr("-v", "Verbose output.");
	Directed = Env.IsArgStr("-dr", "Graph is directed.");
	Weighted = Env.IsArgStr("-w", "Graph is weighted.");
}


void ReadGraph(TStr& InFile, bool& Directed, bool& Weighted, bool& Verbose, PWNet& InNet)
{
	TFIn FIn(InFile);
	int64 LineCnt = 0;
	try {
		while (!FIn.Eof()) {
			TStr Ln;
			FIn.GetNextLn(Ln);
			TStr Line, Comment;
			Ln.SplitOnCh(Line,'#',Comment);
			TStrV Tokens;
			Line.SplitOnWs(Tokens);
			if(Tokens.Len()<2){ continue; }
			int64 SrcNId = Tokens[0].GetInt();
			int64 DstNId = Tokens[1].GetInt();
			double Weight = 1.0;
			if (Weighted) { Weight = Tokens[2].GetFlt(); }
			if (!InNet->IsNode(SrcNId)){ InNet->AddNode(SrcNId); }
			if (!InNet->IsNode(DstNId)){ InNet->AddNode(DstNId); }
			InNet->AddEdge(SrcNId,DstNId,Weight);
			if (!Directed){ InNet->AddEdge(DstNId,SrcNId,Weight); }
			LineCnt++;
		}
		if (Verbose) { printf("Read %lld lines from %s\n", (long long)LineCnt, InFile.CStr()); }
	} catch (PExcept Except) {
		if (Verbose) {
			printf("Read %lld lines from %s, then %s\n", (long long)LineCnt, InFile.CStr(),
			Except->GetStr().CStr());
		}
	}
}


void WriteOutput(TStr& EmbsOutFile, TStr& WalksOutFile, TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV)
{
	TFOut EFOut(EmbsOutFile);
	TFOut WFOut(WalksOutFile);
	for (int64 i = 0; i < WalksVV.GetXDim(); i++) {
		for (int64 j = 0; j < WalksVV.GetYDim(); j++) {
			WFOut.PutInt(WalksVV(i,j));
			if(j+1==WalksVV.GetYDim()) {
				  WFOut.PutLn();
				} else {
				  WFOut.PutCh(' ');
				}
		}
	}
	bool First = 1;
	for (int i = EmbeddingsHV.FFirstKeyId(); EmbeddingsHV.FNextKeyId(i);) {
		if (First) {
			EFOut.PutInt(EmbeddingsHV.Len());
			EFOut.PutCh(' ');
			EFOut.PutInt(EmbeddingsHV[i].Len());
			EFOut.PutLn();
			First = 0;
		}
		EFOut.PutInt(EmbeddingsHV.GetKey(i));
	for (int64 j = 0; j < EmbeddingsHV[i].Len(); j++) {
		EFOut.PutCh(' ');
		EFOut.PutFlt(EmbeddingsHV[i][j]);
	}
	EFOut.PutLn();
	}
}


int main(int argc, char* argv[]) {
	TStr InFile, InFileAttr, EmbsOutFile, WalksOutFile;
	int Dimensions, WalkLen, NumWalks, WinSize, Iter;
	double ParamP, ParamQ, phi;
	bool Directed, Weighted, Verbose, OutputWalks;
	ParseArgs(argc, argv, InFile, InFileAttr, EmbsOutFile, WalksOutFile, Dimensions, WalkLen, NumWalks, WinSize,
		  Iter, Verbose, ParamP, ParamQ, protected_attr, phi, Directed, Weighted);
	PWNet InNet = PWNet::New();
	TIntFltVH EmbeddingsHV, FairEmbeddingsHV;
	TVVec <TInt, int64> WalksVV;

	ReadGraph(InFile, Directed, Weighted, Verbose, InNet);
	node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter, 
		 Verbose, WalksVV, EmbeddingsHV);


	struct n2v_args fairn2v_args = {.net = InNet, .dim = Dimensions, .walklen = WalkLen, .numwalks = NumWalks, .winsize = WinSize, .iter = Iter, .directed = Directed, .weighted = Weighted, .verbose = Verbose};


	// Initialize graph object
	graph g(InFile.CStr(), InFileAttr.CStr()); // Load graph.

	// Check phi.
	phi = (phi == 0) ? g.get_community_percentage(protected_attr) : phi;
	// Create phi file which is needed for phi != r.
	std::ofstream phi_file;
	phi_file.open("communities_phi_share.txt");
	phi_file << "0\t" << 1 - phi << "\n";
	phi_file << "1\t" << phi;
	phi_file.close();
	// Load wanted ratio for categories.
	g.load_community_percentage("communities_phi_share.txt");

	// Initializations.
	srand(time(NULL)); // Seed for rand() from clock.
	int number_of_nodes = g.get_num_nodes(); // Dimension of points.
	
	fairn2v_args.initial_transmtrx = cr_matrix(number_of_nodes, number_of_nodes);
	fairn2v_args.transmtrx = cr_matrix(number_of_nodes, number_of_nodes);
	fairn2v_args.residuals = calc_residuals(g, phi);

	calc_initial_transmtrx(g, fairn2v_args.initial_transmtrx, phi);
	// For iterations.
	std::vector<std::vector<double>> current_point(NUM_POINTS, std::vector<double>(number_of_nodes));
	std::vector<double> candidate_point(number_of_nodes);
	std::vector<double> candidate_direction(number_of_nodes);
	std::vector<double> temp_direction(number_of_nodes);
	double current_loss_function_value[NUM_POINTS];
	double candidate_loss_value;
	double temp_loss_function_value;
	double temp_step;
	int whole_iterations = 0;

	current_point[0] = get_uniform_initial_point(g);
	for(int i=1; i<NUM_POINTS; i++)
		current_point[i] = get_random_initial_point(g);

	for(int i=0; i<NUM_POINTS; i++)
	{
		std::cout << "Calculating initial loss value for point " << i+1 << "\n";
		//Residual Node2Vec
		calc_transmtrx(g, fairn2v_args.transmtrx, fairn2v_args.initial_transmtrx, fairn2v_args.residuals, current_point[i]);
		node2vec(InNet, fairn2v_args.transmtrx, fairn2v_args.dim, fairn2v_args.walklen, fairn2v_args.numwalks, fairn2v_args.winsize, fairn2v_args.iter, fairn2v_args.verbose, fairn2v_args.walks, fairn2v_args.embeddings);
		current_loss_function_value[i] = loss_function_at(g, EmbeddingsHV, fairn2v_args.embeddings, fairn2v_args.dim);
		candidate_loss_value = current_loss_function_value[i];

		for(int j=0; j<MAX_ITERATIONS; j++)
		{
		// Find the best direction and corresponding step.
			for (int k=0; k<NUMBER_OF_DIRECTIONS; k++) {
				std::cout << "Point: " << i+1 << " - " << "Iteration: " << j+1 << " - " << "Direction: " << k+1 << "\n";
				// Get random direction.
				temp_direction = create_random_direction(current_point[i], g);
				// Find feasible best step with bisection.
				// Renew temp_loss_function_value
				temp_step = find_max_step(g, current_point[i], temp_direction, fairn2v_args, EmbeddingsHV, temp_loss_function_value, current_loss_function_value[i]);
				// Check if it is better than the current candidate point.
				if (temp_loss_function_value < candidate_loss_value) {
					// If it is.
					// Renew candidate loss.
					candidate_loss_value = temp_loss_function_value;
					candidate_direction = get_step_direction(temp_step, temp_direction, number_of_nodes);
					// Renew the candidate point.
					get_candidate_point(current_point[i], candidate_direction, candidate_point, number_of_nodes);
					//save_results(temp_fair_pagerank, current_point);
				}
			}
		}

		// Renew values.
		current_point[i] = candidate_point;
		current_loss_function_value[i] = candidate_loss_value;
		std::cout << "Point " << i+1 << " minimized loss value: " << current_loss_function_value[i] << std::endl;
		
	}

	int result_point_index = std::distance(current_loss_function_value, std::min_element(current_loss_function_value, current_loss_function_value + (NUM_POINTS-1)));
	std::vector<double> result_point = current_point[result_point_index];

	//Residual Node2Vec
	calc_transmtrx(g, fairn2v_args.transmtrx, fairn2v_args.initial_transmtrx, fairn2v_args.residuals, result_point);
	node2vec(InNet, fairn2v_args.transmtrx, fairn2v_args.dim, fairn2v_args.walklen, fairn2v_args.numwalks, fairn2v_args.winsize, fairn2v_args.iter, fairn2v_args.verbose, fairn2v_args.walks, fairn2v_args.embeddings);

	std::cout << "Global minimized loss value: " << current_loss_function_value[result_point_index] << std::endl;
	WriteOutput(EmbsOutFile, WalksOutFile, fairn2v_args.embeddings, fairn2v_args.walks);
	
	return 0;
}
