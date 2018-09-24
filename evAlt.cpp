//included all essentials I think
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>
#include <string>
#include <vector>

int main (int argc, char *argv[]) {

	//This is where we will also read in the file it just needs some work right now
	//just wanted to give a little head start to the project



	//Thinking about creating an if statement to determine if its ga or pbil.
	string filename = argv[1];
	int num_individuals = atoi(argv[2]);
	string crossover = argv[3];
	int cross_prop = atoi(argv[4]);
	int mut_prop = atoi(argv[5]);
	int num_gen = atoi(argv[6]);
	string alg = argv[7];


	//once we are done calculating we have to provide output to a file
	//Also need to keep track of best output so far

}



class Individual{

	//are we creating a vector to hold the clauses
	public:
		int fitness;
		int calcFitness();



}

//probably need to talk about this data structure to make sure we are doing this right
vector< vector<pair<bool, bool>> >readFile(){

}

int Individual::calcFitness(){

	//calc and return integer for fitness
}
//dont know whether we wanted to do this or not
class Population{


}

//dont know what this will return yet
void rank_selection(vector<Individual> Population){

	//manipulate population here or as a global.



}

void tournament_selection(vector<Individual> Population){

	//manipulate population here or as a global.
	


}


void boltzman_selection(vector<Individual> Population){



}



int genetic_alg(string slection_type, int num_individuals, int cross_prop, int mut_prop, int num_gen){
	//have seperate functions for each selection type and run a vertain amount of times



}


int pbil(int num_individuals, int pos_learning_rate, int neg_learning_rate, int mut_prop, num_gen){

	//do pbil with population from here
}





