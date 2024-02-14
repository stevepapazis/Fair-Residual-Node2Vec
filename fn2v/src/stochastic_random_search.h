#ifndef STOCHASTIC_RND_SEARCH_H
#define STOCHASTIC_RND_SEARCH_H

#include "fn2v.h"
#include "graph.hpp"
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <algorithm>
#include <set>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <math.h>


extern int MAX_ITERATIONS;
extern int NUM_POINTS;
extern int NUMBER_OF_DIRECTIONS;
extern double PRECISION_OF_SOLUTION;
extern double PRECISION_OF_CAT_RATIO;
extern int TIMES_TO_TRY_FOR_INITIAL_POINT;
extern double INITIAL_STEP;
extern int SMALLER_ALLOWED_STEP;

extern int protected_attr;

extern TIntFltVH embeddings;

struct n2v_args;


double get_euklidean_norm(std::vector<double> &vec, int dimension);
std::vector<double> get_step_direction(double step, std::vector<double> &direction, int dimension);
std::vector<double> get_random_initial_point(graph &g);
std::vector<double> get_uniform_initial_point(graph &g);
std::vector<double> create_random_direction(std::vector<double> &current_point, graph &g);
double find_max_step(graph &g, std::vector<double> &current_point, std::vector<double> &direction, struct n2v_args, TIntFltVH &embeddings, double &temp_loss_function_value, double &current_loss_function_value);
double loss_function_at(graph&, TIntFltVH &embeddings, TIntFltVH &fair_embeddings, int dim);
bool is_probability_vector(std::vector<double> &point, graph &g);
void get_candidate_point(std::vector<double> &current_point, std::vector<double> &step_direction, std::vector<double> &point, int dimension);
bool is_reduction_direction(double current_loss, double temp_loss);


#endif
