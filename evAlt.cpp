//included all essentials I think
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>
#include <random>
#include <string>
#include <string.h>
#include <vector>
#include <sstream>
using namespace std;




class Individual{

	//are we creating a vector to hold the clauses
	public:
		int fitness;
		int calcFitness();
		// feel free to rename later - KP
		vector<int> negationArray;



};

//probably need to talk about this data structure to make sure we are doing this right
vector< vector<int> > readFile(string name){

	cout << "here" << endl;
	vector< vector<int> > clauseFile;
	vector<int> clause;

	ifstream input;
	string line;

	input.open(name);
	getline(input, line);
	if(! input.is_open()){
		cout << "errror opening file" << endl;
        return clauseFile;
	}
	
	while(getline(input, line)){
		string delimiter = " ";
		int position;
		string number;

		while ((position = line.find(delimiter)) != string::npos){
   			number = line.substr(0, position);

   			clause.push_back(stoi(number));
    		cout << stoi(number) << endl;
    		line.erase(0, position + delimiter.length());
		}


		clauseFile.push_back(clause);
		clause.clear();

	}

	return clauseFile;


}

// Changed this from int to void -- let me know if you think this still needs to be an int.
void Individual::calcFitness(vector<vector<int>> clauseFile){

	for(int i = 0; i < clauseFile.size(); i++) {
		for(int j = 0; j < clauseFile.at(i).size(); j++){
			if(clauseFile.at(i).at(j) == 0)
				break;

			else if(clauseFile.at(i).at(j) * negationArray.at(clauseFile.at(i).at(j)-1) > 0) {
				fitness++;
				break;
			}
		}
	}


}
//dont know whether we wanted to do this or not
class Population{


};

//dont know what this will return yet
void rank_selection(vector<Individual> Population){

	//manipulate population here or as a global.



}


void tournament_selection(vector<Individual> Population){

	//manipulate population here or as a global.
	


}


void boltzman_selection(vector<Individual> Population){



}

void mutate(double mutProb, Individual individual) {
	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;

	for(int i = 0; i < individual.negationArray.size(); i++){
		// randDouble(randomEngine) returns a random double between 0 and 1
		if( randDouble(randomEngine) <= mutProb ) 
			individual.negationArray.at(i) = individual.negationArray.at(i) * -1;

	}

}
// Takes in two parent inviduals and the crossover probability
// Returns a pair of children.
// Haven't tested this function as of yet. 
pair<Individual, Individual> onePointCrossover(double crossProb, Individual firstParent, Individual secondParent) {

	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;

   	Individual firstChild, secondChild;

   	int crossoverPointIndex = 2 * ( rand() % ((firstParent.negationArray.size()-2)/2) ) + 1;

   	cout << "crossoverPointIndex: " << crossoverPointIndex << endl;

   	for(int i = 0; i < firstParent.negationArray.size(); i++) {
   		if(i > crossoverPointIndex)
   			firstChild.negationArray.push_back(secondParent.negationArray.at(i));
   		else
   			firstChild.negationArray.push_back(firstParent.negationArray.at(i));
   	}

   	for(int j = 0; j > 0; j++) {
   		if(j < crossoverPointIndex)
   			secondChild.negationArray.push_back(secondParent.negationArray.at(j));
   		else
   			secondChild.negationArray.push_back(firstParent.negationArray.at(j));
   	}
}

Individual uniformCrossover(double crossProb, Individual firstParent, Individual secondParent) {

	Individual child;
	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;


	for(int i = 0; i < firstParent.negationArray.size(); i++) {
		if(randDouble(randomEngine) <= .5) {
			child.negationArray.push_back(firstParent.negationArray.at(i));
		}
		else{
			child.negationArray.push_back(secondParent.negationArray.at(i));
		}
	}

	return child;

}


int genetic_alg(string slection_type, int num_individuals, double crossProb, double mutProb, int num_gen){
	//have seperate functions for each selection type and run a vertain amount of times



}


int pbil(int num_individuals, int pos_learning_rate, int neg_learning_rate, double mutProb, int num_gen){

	//do pbil with population from here
}


int main (int argc, char *argv[]) {

	srand (time(NULL));
	for(int i = 0; i < 50; i++ ){
		cout << "crossoverPointIndex: " << 2 * ( rand() % ((20-2)/2) ) + 1 << endl;

	}

	vector<vector<int> > clauseFile;

	//This is where we will also read in the file it just needs some work right now
	//just wanted to give a little head start to the project



	//Thinking about creating an if statement to determine if its ga or pbil.
	string filename = argv[1];
	// int num_individuals = atoi(argv[2]);
	// string crossover = argv[3];
	// double crossProb = atoi(argv[4]);
	// double mutProb = atoi(argv[5]);
	// int num_gen = atoi(argv[6]);
	// string alg = argv[7];


	//once we are done calculating we have to provide output to a file
	//Also need to keep track of best output so far

	clauseFile = readFile(filename);

}



