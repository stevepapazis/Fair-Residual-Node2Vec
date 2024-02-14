#include "stdafx.h"
#include "n2v.h"
#include <fstream>
#include <omp.h>
#include <chrono>
#include <iomanip>


void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& InFileAttr, TStr& EmbsOutFile, TStr& WalksOutFile, int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose, double& ParamP, double& ParamQ, bool& Directed, bool& Weighted)
{
	Env = TEnv(argc, argv, TNotify::StdNotify);
	Env.PrepArgs(TStr::Fmt("\nAn algorithmic framework for representational learning on graphs."));
	InFile = Env.GetIfArgPrefixStr("-i:", "graphs/fb_edges.txt", "Input graph path");
	InFileAttr = Env.GetIfArgPrefixStr("-iattr:", "graphs/fb_genders.txt", "Input attributes path");
	EmbsOutFile = Env.GetIfArgPrefixStr("-oe:", "graphs/fb_n2v_embs.txt", "Output graph embs path");
	WalksOutFile = Env.GetIfArgPrefixStr("-ow:", "graphs/fb_n2v_walks.txt", "Output graph walks path");
	Dimensions = Env.GetIfArgPrefixInt("-d:", 128, "Number of dimensions. Default is 128");
	WalkLen = Env.GetIfArgPrefixInt("-l:", 80, "Length of walk per source. Default is 80");
	NumWalks = Env.GetIfArgPrefixInt("-r:", 10, "Number of walks per source. Default is 10");
	WinSize = Env.GetIfArgPrefixInt("-k:", 10, "Context size for optimization. Default is 10");
	Iter = Env.GetIfArgPrefixInt("-e:", 1, "Number of epochs in SGD. Default is 1");
	ParamP = Env.GetIfArgPrefixFlt("-p:", 1, "Return hyperparameter. Default is 1");
	ParamQ = Env.GetIfArgPrefixFlt("-q:", 1, "Inout hyperparameter. Default is 1");
	Verbose = Env.IsArgStr("-v", "Verbose output.");
	Directed = Env.IsArgStr("-dr", "Graph is directed.");
	Weighted = Env.IsArgStr("-w", "Graph is weighted.");
	//OutputWalks = Env.IsArgStr("-ow", "Output random walks instead of embeddings.");
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
	double ParamP, ParamQ;
	bool Directed, Weighted, Verbose, OutputWalks;
	ParseArgs(argc, argv, InFile, InFileAttr, EmbsOutFile, WalksOutFile, Dimensions, WalkLen, NumWalks, WinSize,
		  Iter, Verbose, ParamP, ParamQ, Directed, Weighted);
	PWNet InNet = PWNet::New();
	TIntFltVH EmbeddingsHV, FairEmbeddingsHV;
	TVVec <TInt, int64> WalksVV;

	ReadGraph(InFile, Directed, Weighted, Verbose, InNet);
	node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter, Verbose, WalksVV, EmbeddingsHV);
	WriteOutput(EmbsOutFile, WalksOutFile, EmbeddingsHV, WalksVV);

	return 0;
}
