#include "stochastic_random_search.h"


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


bool is_reduction_direction(double current_loss, double temp_loss) {
    if (temp_loss < current_loss) {
        return true;
    } else {
        return false;
    }
}

void get_candidate_point(std::vector<double> &current_point, std::vector<double> &step_direction, std::vector<double> &point, int dimension) {
    for (int i = 0; i < dimension; i++) {
        point[i] = current_point[i] + step_direction[i];
    }
}

std::vector<double> get_step_direction(double step, std::vector<double> &direction, int dimension) {
    std::vector<double> step_direction(dimension);
    for (int i = 0; i < dimension; i++) {
        step_direction[i] = step * direction[i];
    }

    return step_direction;
}

std::vector<double> get_random_initial_point(graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    double total_quantity = 1;
    double total_quantity_to_red = 1;
    double total_quantity_to_blue = 1;
    double quantity_to_give = 0;
    double max_quantity_to_give = 1;
    std::vector<int> nodes(dimension);
    for (int i = 0; i < dimension; i ++) {
        nodes[i] = i;
    }
    std::random_shuffle(nodes.begin(), nodes.end());

    int node = 0;
    while (total_quantity > pow(10, -4)) {
        if (g.get_community(nodes[node]) == 1) {
            max_quantity_to_give = std::min(total_quantity_to_red, 1 - initial_point[nodes[node]]);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_quantity_to_give;
            initial_point[nodes[node]] += quantity_to_give;
            total_quantity_to_red -= quantity_to_give;    
        } else {
            max_quantity_to_give = std::min(total_quantity_to_blue, 1 - initial_point[nodes[node]]);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_quantity_to_give;
            initial_point[nodes[node]] += quantity_to_give;
            total_quantity_to_blue -= quantity_to_give;
        }
        total_quantity -= quantity_to_give;
        node ++;
        if (node == dimension) {
            node = 0;
        } 
    }
    for (int i = 0; i < dimension; i++) {
        if ((1 - initial_point[nodes[i]] > total_quantity_to_red) && (g.get_community(nodes[i]) == 1)) {
            initial_point[nodes[i]] += total_quantity_to_red;
            break;
        }
    }
    for (int i = 0; i < dimension; i++) {
        if ((1 - initial_point[nodes[i]] > total_quantity_to_blue) && (g.get_community(nodes[i]) == 0)) {
            initial_point[nodes[i]] += total_quantity_to_blue;
            break;
        }
    }
    //std::cout << "Initailized\n";
    
    double red_sum = 0;
    double blue_sum = 0;
    for (int i = 0; i < dimension; i++) {
        if (g.get_community(i) == 1) {
            red_sum += initial_point[i];
        } else {
            blue_sum += initial_point[i];
        }
    }
    //std::cout << "Red ratio: " << red_sum << "blue ratio: " << blue_sum << "\n";
    

    return initial_point;
}

// Returns the unform excess policy.
std::vector<double> get_uniform_initial_point(graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    for (int i = 0; i < g.get_num_nodes(); i++) {
        if (g.get_community(i) == 0) {
            initial_point[i] = 1 / (double)g.get_community_size(0);
        } else {
            initial_point[i] = 1 /(double)g.get_community_size(1);
        }
    }
    return initial_point;
}

std::vector<double> create_random_direction(std::vector<double> &current_point, graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double> direction(dimension, 0);
    std::vector<double> temp_point(dimension);
    std::vector<int> nodes(dimension);
    double quantity_to_add = 0;
    double quantity_to_add_red = 0;
    double quantity_to_add_blue = 0;
    double node_quantity = 0;
    double quantity_to_take = 0;
    double quantity_to_give = 0;
    double max_to_give;
    int node;

    for (int i = 0; i < dimension; i ++) {
        nodes[i] = i;
    }

    for (int i = 0; i < dimension; i++) {
        node_quantity = current_point[i];
        if (node_quantity > 0) {
            quantity_to_take = ((double)rand() / RAND_MAX) * node_quantity;
            direction[i] -= quantity_to_take;
            if (g.get_community(i) == 1) {
                quantity_to_add_red += quantity_to_take;
            } else {
                quantity_to_add_blue += quantity_to_take;
            }
            quantity_to_add += quantity_to_take;
        }
    }
  
    std::random_shuffle(nodes.begin(), nodes.end());
    get_candidate_point(current_point, direction, temp_point, dimension);

    int i = 0;
    while (quantity_to_add > pow(10, -4)) {
        node = nodes[i];
        if (g.get_community(node) == 1) {
            max_to_give = std::min((1-temp_point[i]), quantity_to_add_red);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_to_give;
            direction[node] += quantity_to_give;
            quantity_to_add_red -= quantity_to_give;
        } else {
            max_to_give = std::min((1-temp_point[i]), quantity_to_add_blue);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_to_give;
            direction[node] += quantity_to_give;
            quantity_to_add_blue -= quantity_to_give;
        }
        quantity_to_add -= quantity_to_give;
        i++;
        if (i == dimension) {
            i = 0;
            get_candidate_point(current_point, direction, temp_point, dimension);
        }
    }

    std::random_shuffle(nodes.begin(), nodes.end());
    for (int i = 0; i < dimension; i ++) {
        node = nodes[i];
        if (1 - temp_point[node] > quantity_to_add) {
            if (g.get_community(i) == 1) {
                direction[node] += quantity_to_add_red;
                quantity_to_add_red = 0;
                quantity_to_add -= quantity_to_add_red;
            } else {
                direction[node] += quantity_to_add_blue;
                quantity_to_add_blue = 0;
                quantity_to_add -= quantity_to_add_blue;
            }
        }
    }
    get_candidate_point(current_point, direction, temp_point, dimension);
    double red_sum = 0;
    double blue_sum = 0;
    for (int i = 0; i < dimension; i++) {
        if (g.get_community(i) == 1) {
            red_sum += temp_point[i];
        } else {
            blue_sum += temp_point[i];
        }
    };

    return direction;
}

bool is_valid_residual_policy(std::vector<double> &point, graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double>::iterator counter;
    double red_sum = 0;
    double blue_sum = 0;

    for (int i = 0; i < dimension; i++) {
        // Sums each category's probabilities.
        (g.get_community(i) == 1) ? red_sum += point[i] : blue_sum += point[i];
        // Checks for coordinates to have the sense of probability.
        if (point[i] < 0 || point[i] > 1) {
            return false;
        }
    }

    // Checks for each category's probabilities to have the
    // sense of probability distribution.
    if (abs(red_sum - 1) > PRECISION_OF_CAT_RATIO || abs(blue_sum - 1) > PRECISION_OF_CAT_RATIO) {
        return false;
    }

    return true;
}

double find_max_step(graph &g, std::vector<double> &current_point, std::vector<double> &direction, struct n2v_args fairn2v_args, TIntFltVH &embeddings, double &temp_loss_function_value, double &current_loss_function_value) 
{
    int dimension = g.get_num_nodes();
    std::vector<double> point(dimension);
    std::vector<double> step_direction(dimension);
    //pagerank_v temp_pagerank;
    TIntFltVH temp_fair_embeddings;
    bool change_sign = true;

    // Find reduction direction.
    double step = INITIAL_STEP;
    // Return step * direction.
    step_direction = get_step_direction(step, direction, dimension);
    // Stores at <point> the candidate_point = current + step_direction.
    get_candidate_point(current_point, step_direction, point, dimension);
    // Start time for loss calculation.
    //loss_start_time = std::chrono::high_resolution_clock::now();
    
    //Residual Node2Vec
    calc_transmtrx(g, fairn2v_args.transmtrx, fairn2v_args.initial_transmtrx, fairn2v_args.residuals, point);
    node2vec(fairn2v_args.net, fairn2v_args.transmtrx, fairn2v_args.dim, fairn2v_args.walklen, fairn2v_args.numwalks, fairn2v_args.winsize, fairn2v_args.iter, fairn2v_args.verbose, fairn2v_args.walks, fairn2v_args.embeddings);
    //calc_transmtrx(g, transmtrx, initial_transmtrx, residuals, current_point);
    //node2vec(InNet, transmtrx, Dimensions, WalkLen, NumWalks, WinSize, Iter, Verbose, OutputWalks, WalksVV, temp_fair_embeddings);
    temp_loss_function_value = loss_function_at(g, embeddings, fairn2v_args.embeddings, fairn2v_args.dim);

    // Get temp custmo LFPR and its loss value at candidate_point.
    //temp_pagerank = algs.get_custom_step_fair_pagerank(point);
    //temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);

    // Find step (alogn with sign) for reduction direction.
    while(!is_reduction_direction(current_loss_function_value, temp_loss_function_value)) {
        if (change_sign) {
            step = - step;
        } else {
            step = step/(double)2;
        }
        change_sign = !change_sign;
	std::cout << "Step: " << step << std::endl;

        // Same as before.
        step_direction = get_step_direction(step, direction, dimension);
        get_candidate_point(current_point, step_direction, point, dimension);
        // Start time for loss calculation.
        //loss_start_time = std::chrono::high_resolution_clock::now();

        //Residual Node2Vec
	calc_transmtrx(g, fairn2v_args.transmtrx, fairn2v_args.initial_transmtrx, fairn2v_args.residuals, point);
        node2vec(fairn2v_args.net, fairn2v_args.transmtrx, fairn2v_args.dim, fairn2v_args.walklen, fairn2v_args.numwalks, fairn2v_args.winsize, fairn2v_args.iter, fairn2v_args.verbose, fairn2v_args.walks, fairn2v_args.embeddings);
        //calc_transmtrx(g, transmtrx, initial_transmtrx, residuals, current_point);
        //node2vec(InNet, transmtrx, Dimensions, WalkLen, NumWalks, WinSize, Iter, Verbose, OutputWalks, WalksVV, temp_fair_embeddings);
        temp_loss_function_value = loss_function_at(g, embeddings, fairn2v_args.embeddings, fairn2v_args.dim);

        // Get temp custmo LFPR and its loss value at candidate_point.
        //temp_pagerank = algs.get_custom_step_fair_pagerank(point);
        //temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);

        // If step is smaller than allowed, return 0 step
        // (i.e. stay at the same point).
        if (abs(step) < SMALLER_ALLOWED_STEP) {
            //std::cout << "zero step";
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);

	    calc_transmtrx(g, fairn2v_args.transmtrx, fairn2v_args.initial_transmtrx, fairn2v_args.residuals, point);
            node2vec(fairn2v_args.net, fairn2v_args.transmtrx, fairn2v_args.dim, fairn2v_args.walklen, fairn2v_args.numwalks, fairn2v_args.winsize, fairn2v_args.iter, fairn2v_args.verbose, fairn2v_args.walks, fairn2v_args.embeddings);
            temp_loss_function_value = loss_function_at(g, embeddings, fairn2v_args.embeddings, fairn2v_args.dim);

            //temp_pagerank = algs.get_custom_step_fair_pagerank(point);
            // Important to renew the temp_loss_function_value.
            // Not in this control, but keep it for cohesion.
            //temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);

            return step;
        }
    }

    // Find valid residual policy.
    if (step < 0) {
        while (!is_valid_residual_policy(point, g) && abs(step) > SMALLER_ALLOWED_STEP) {
            step = step/(double)2;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
	std::cout << "Step: " << step << std::endl;
        }
        if (abs(step) < SMALLER_ALLOWED_STEP) {
            //std::cout << "zero step";
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        calc_transmtrx(g, fairn2v_args.transmtrx, fairn2v_args.initial_transmtrx, fairn2v_args.residuals, point);
        node2vec(fairn2v_args.net, fairn2v_args.transmtrx, fairn2v_args.dim, fairn2v_args.walklen, fairn2v_args.numwalks, fairn2v_args.winsize, fairn2v_args.iter, fairn2v_args.verbose, fairn2v_args.walks, fairn2v_args.embeddings);
        temp_loss_function_value = loss_function_at(g, embeddings, fairn2v_args.embeddings, fairn2v_args.dim);

        //temp_pagerank = algs.get_custom_step_fair_pagerank(point);
        //temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    }

    return step;
}

double loss_function_at(graph &g, TIntFltVH &embeddings, TIntFltVH &fair_embeddings, int dimension) {
	node_t *nodes = g.get_nodes();
	int nnodes = g.get_num_nodes();
	double sum = 0;
	double distance = 0;

	for (int i = 0; i < nnodes; i++) {
		distance = 0;
		for(int j=0; j<dimension; j++)
			distance += pow((embeddings[nodes[i].id][j] - fair_embeddings[nodes[i].id][j]), 2);
		sum += sqrt(distance);
	}

    return sum;
}
