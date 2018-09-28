//included all essentials I think
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>
#include <random>
#include <string>
#include <string.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <random>
#include <math.h>
using namespace std;

const int TOURNAMENT_SELECTION_M = 2;
const int TOURNAMENT_SELECTION_K = 1;

int numberOfVariables;

class Individual{

	//are we creating a vector to hold the clauses
	public:
		vector<int> varAssignmentArray;
		int fitness = 0;
		void calcFitness(vector<vector<int>> clauseFile);

};

// Changed this from int to void -- let me know if you think this still needs to be an int.
void Individual::calcFitness(vector<vector<int>> clauseFile){

	for(int i = 0; i < clauseFile.size(); i++) {
		for(int j = 0; j < clauseFile.at(i).size(); j++){

			if(clauseFile.at(i).at(j) * this->varAssignmentArray.at(clauseFile.at(i).at(j)-1) > 0) {
				this->fitness++;
				break;
			}
		}
	}

}

//probably need to talk about this data structure to make sure we are doing this right
vector< vector<int> > readFile(string name){

	//Data structure used to keep track of the whole fike
	vector< vector<int> > clauseFile;

	//each individual clause
	vector<int> clause;

	ifstream input;
	string line;

	//open file and get rid of first line
	input.open(name);
	getline(input, line);
	cout << line << endl;
	string delimiter = " ";
	int position;
	string number;
	for(int i = 0; i < 3; i++){

		position = line.find(delimiter);
		if(position == string::npos){
			cout << "error" << endl;
		}

		number = line.substr(0, position);

		line.erase(0, position + delimiter.length());

	}
	numberOfVariables = stoi(number);

	//check to see if the file is actually open
	if(! input.is_open()){
		cout << "errror opening file" << endl;
        return clauseFile;
	}

	//while the file is still open.
	while(getline(input, line)){

		//split string by spaces
		string delimiter = " ";

		//position of delimeter
		int position;
		string number;

		//while there is still space and we didnt not find anything
		//this gets rid of the zero at the end of the clasues.
		while ((position = line.find(delimiter)) != string::npos){
   			number = line.substr(0, position);
   			
   			//push back variable into clause vector
   			clause.push_back(stoi(number));
   			//move on to other variables in the line
    		line.erase(0, position + delimiter.length());
		}
		
		//push back clause into big vector
		clauseFile.push_back(clause);

		//be safe and clear vector
		clause.clear();

	}

	//return completed vector
	return clauseFile;

}

bool sortPopulation(const Individual & s1, const Individual & s2){
   return s1.fitness < s2.fitness;
}

vector<Individual> rankSelection(vector<Individual> population, int numIndividuals){

	sort(population.begin(), population.end(), sortPopulation);
	vector<int> rankProbabilities;
	vector<Individual> breedingPool;
	int rankSum = numIndividuals * (numIndividuals + 1) * 0.5;

	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> gen(1, rankSum); // uniform, unbiased

	for(int i = 0; i < numIndividuals; i++){
		if(i == 0){
			rankProbabilities.push_back(i + 1);
		}
		else {
			rankProbabilities.push_back(i + 1 + rankProbabilities.at(i-1));
		}
	}


	for(int i = 0; i < numIndividuals; i++){
		int rand = gen(engine);
		cout << rand << endl;
		for(int j = 0; j < numIndividuals; j++){
			if(rand <= rankProbabilities.at(j)){
				breedingPool.push_back(population.at(j));
				break;
			}
		}
	}

	return breedingPool;

}


vector<Individual> tournamentSelection(vector<Individual> Population, int numIndividuals){

	sort(population.begin(), population.end(), sortPopulation);
	vector<Individual> breedingPool;

	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> gen(0, numIndividuals-1); // uniform, unbiased

	/* conducts tournament selection by choosing best m individuals (m random numbers)
       and then choosing the best fit individual (highest number since individuals are
       sorted by fitness) 
     */
	for(int i = 0; i < numIndividuals; i++){
		vector<int> tournamentM;
		for(int j = 0; j < TOURNAMENT_SELECTION_M; j++){
			tournamentM.push_back(gen(engine));
		}
		breedingPool.push_back(population.at(*max_element(tournamentM.begin(), tournamentM.end())));
	}

	return breedingPool;

}


vector<Individual> boltzmannSelection(vector<Individual> Population, int numIndividuals){
	vector<Individual> breedingPool;
	double sumBolztmannProbabilites = 0;

	for(int i = 0; i < numIndividuals; i++){
		sumBolztmannProbabilites += exp (population.at(i).fitness);
	}

	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_real_distribution<double> gen(0.0, 1.0);

	vector<double> boltzmannProbabilities;

	for(int i = 0; i < numIndividuals; i++){
		if(i == 0){
			boltzmannProbabilities.push_back((exp (population.at(i).fitness)) / sumBolztmannProbabilites);
		}
		else {
			boltzmannProbabilities.push_back(((exp (population.at(i).fitness)) / sumBolztmannProbabilites) 
				+ boltzmannProbabilities.at(i-1));
		}
	}

	for(int i = 0; i < numIndividuals; i++){
		double rand = gen(engine);
		for(int j = 0; j < numIndividuals; j++){
			if(rand <= boltzmannProbabilities.at(j)){
				breedingPool.push_back(population.at(j));
				break;
			}
		}
	}

	return breedingPool;

}


vector<Individual> mutatePopulation(vector<Individual> population, double mutProb, int numIndividuals) {
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_real_distribution<double> gen(0.0, 1.0);

   	for(int i = 0; i < numIndividuals; i++){
   		for(int j = 0; j < numberOfVariables; j++){
			if(gen(engine) < mutProb) 
				population.at(i).varAssignmentArray.at(j) *= -1;
		}
   	}

   	return population;	

}

// Takes in two parent inviduals and the crossover probability
// Returns a pair of children.
// Haven't tested this function as of yet. 
vector<Individual> onePointCrossover(vector<Individual> population, double crossProb, int numIndividuals){

	std::random_device seeder;
	std::mt19937 engine(seeder());
	//we want to choose a crossover point (that excludes the ends) so we go from 1 to numVariables - 1
	std::uniform_int_distribution<int> gen(1, numberOfVariables - 1);
	std::uniform_int_distribution<double> genDouble(0.0, 1.0);


   	Individual firstChild, secondChild;
   	vector<Individual> newPopulation;

   	for(int i = 0; i < numIndividuals; i += 2) {

   		//checks whether crossover happens
   		if(genDouble(engine) <= crossProb){
   			int crossoverPointIndex = gen(engine);

   			//1-point crossover occurs at random index and creates two children
   			for(int j = 0; j < numberOfVariables; j++) {
   				if(j >= crossoverPointIndex) {
   					firstChild.varAssignmentArray.push_back(population.at(i+1).varAssignmentArray.at(j));
   					secondChild.varAssignmentArray.push_back(population.at(i).varAssignmentArray.at(j));
   				}
   				else {
   					firstChild.varAssignmentArray.push_back(population.at(i).varAssignmentArray.at(j));
   					secondChild.varAssignmentArray.push_back(population.at(i+1).varAssignmentArray.at(j));
   				}
   			}

   			// Add children to return population and clear them for the next set of crossover
   			newPopulation.push_back(firstChild);
   			newPopulation.push_back(secondChild);
   			firstChild.varAssignmentArray.clear();
   			secondChild.varAssignmentArray.clear();
   		}

   		//if crossover doesn't happen, we keep the individuals from the old population
   		else {
   			newPopulation.push_back(population.at(i));
			newPopulation.push_back(population.at(i+1));
   		}
   		
   	}

   	return newPopulation;

   	
}

vector<Individual> uniformCrossover(vector<Individual> population, double crossProb, int numIndividuals) {

	std::random_device seeder;
	std::mt19937 engine(seeder());
	//we want to choose a crossover point (that excludes the ends) so we go from 1 to numVariables - 1
	std::uniform_int_distribution<int> gen(1, numberOfVariables - 1);
	std::uniform_int_distribution<double> genDouble(0.0, 1.0);


	Individual firstChild, secondChild;
   	vector<Individual> newPopulation;

   	for(int i = 0; i < numIndividuals; i += 2){

   		//checks whether crossover happens
   		if(genDouble(engine) <= crossProb) {
   			for(int j = 0; j < numberOfVariables; j++) {
   				
   				//creates the first child
				if(genDouble(engine) <= .5) {
					firstChild.varAssignmentArray.push_back(population.at(i).varAssignmentArray.at(j));
				}
				else {
					firstChild.varAssignmentArray.push_back(population.at(i+1).varAssignmentArray.at(j));
				}

				/* Creates the second child. Since we want the new population size to match old population
			  	   size and uniform crossover only makes 1 child, we will do two crossovers for each pair of
			       parents (gives us two children from each parent).
				*/
				if(genDouble(engine) <= .5) {
					secondChild.varAssignmentArray.push_back(population.at(i).varAssignmentArray.at(j));
				}
				else {
					secondChild.varAssignmentArray.push_back(population.at(i+1).varAssignmentArray.at(j));
				}


			}

			newPopulation.push_back(firstChild);
			newPopulation.push_back(secondChild);
			firstChild.varAssignmentArray.clear();
			secondChild.varAssignmentArray.clear();
   		}

   		//if crossover doesn't happen, we keep the individuals from the old population
   		else {
   			newPopulation.push_back(population.at(i));
			newPopulation.push_back(population.at(i+1));
   		}

   	}

	return newPopulation;
}


int genetic_alg(string selectionType, string crossoverType, int numberOfClauses, int numIndividuals, double crossProb, double mutProb, int numGen){
	vector<Individual> population;
	// Initialize population here
	int genCount = 0;

	//have seperate functions for each selection type and run a vertain amount of times
	while(genCount != numGen || population.at(0).fitness != numberOfClauses) {
		//does selection based on user input
		if(selectionType == "rs")
			population = rankSelection(population, numIndividuals);
		else if(selectionType == "ts")
			population = tournamentSelection(population, numIndividuals);
		else if (selectionType == "bs")
			population = boltzmannSelection(population, numIndividuals);


		//does crossover based on user input
		if(crossoverType == "1c") 
			population = onePointCrossover(population, crossProb, numIndividuals);
		else if(crossoverType == "uc")
			population = uniformCrossover(population, crossProb, numIndividuals);

		population = mutatePopulation(population, mutProb, numIndividuals);

		genCount++;
	}

}



//returns index of worst solution
bestSolution findBestSolution(vector<Individual> sampleVector, vector<int> evaluations, bestSolution best){

	int bestFitness = 0;
	int bestVectorIndex = 0;

	for(int i = 0; i < sampleVector.size(); i++){
		if(evaluations[i] > bestFitness){
			bestFitness = evaluations[i];
			bestVectorIndex = i;
		}
	}

	best.bestVector = sampleVector[bestVectorIndex].negationArray;
	best.bestFitness = bestFitness;

}

//returns index of best solution
vector<int> findWorstSolution(vector<Individual> sampleVector, vector<int> evaluations){


	int worstFitness = 8000;
	int worstFitnessIndex = 0;

	for(int i = 0; i < sampleVector.size(); i++){
		if(evaluations[i] < worstFitness){
			worstFitness = evaluations[i];
			worstFitnessIndex = i;
		}
	}

	return sampleVector[worstFitnessIndex].negationArray;

}


//generates a vector of bools for use in a pbil
Individual* generateSampleVector(vector<double> probVector, int numberOfClauses){

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Individual *indv = new Individual;


	for(int i = 0; i < probVector.size(); i ++){

		//probability for chosing true or false;
		double probForUse = dis(gen);

		//if the porbability is useful than its one
		if(probForUse <= probVector[i]){
			indv->negationArray[i] = 1;
		}

		//if the probability is not than its zero
		else{
			indv->negationArray[i] = 0;
		}
	}

	return indv;
}


int evaluate(Individual indv, vector<vector<int> > clauseFile){

	indv.calcFitness(clauseFile);
	return indv.fitness;

}


int pbil(vector<vector<int> > clauseFile, int numberOfClauses, int numIndividuals, int posLearningRate, int negLearningRate, double mutProb, int numGen){


	//helps us keep track of best individual so far
	struct bestSolutionSoFar
	{
		vector<int> bestVector;
		int bestFitness;
		
	};

	//create struct to keep track of bestIndividual
	bestSolutionSoFar bestIndividual;
	

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
	
    //used to keep track of fitness for vectors
	vector<int> evaluations;

	//probability vector
	vector<double> probVector;


	//best vector we have found so far
	vector<int> bestVectorSoFar;
	//the fitness of best vector
	int bestFitnessSoFar = 0;

	//vector used for sample population
	vector<Individual> sampleVector;

	//set normal probabilities
	for(int i =0; i < numberOfVariables; i++){
		probVector[i] = 0.5;
	}

	//generation nuymber
	int generation = 0;
	
	while(generation < numGen){
		
		for(int i = 0; i < numIndividuals; i ++){
			sampleVector[i] = *generateSampleVector(probVector, numberOfClauses);

			//pretty unsure about this at the moment
			evaluations[i] = evaluate(sampleVector[i], clauseFile);
		}

		vector<int> bestVector;
		vector<int> worstVector;

		//pass data into struct that keeps track of best solutions
		bestIndividual = findBestSolution(sampleVector, evaluations, bestIndividual);

		//gives us the best vector from this iteration
		bestVector = bestIndividual.bestVector;

		//keep track of best individual we have found
		if(bestIndividual.fitness > bestFitnessSoFar){
			bestFitnessSoFar = bestIndividual.fitness;
			bestVectorSoFar = bestVector;
		}

		//keep track of worst vector for the increment
		worstVector = findWorstSolution(sampleVector, evaluations);

		//generate positive learning rate 
		for(int i = 0; i < probVector.size(); i++){
			probVector[i] = probVector[i] * (1.0 - posLearningRate) + (bestVector[i] * posLearningRate);

		}

		//generate nefatice learning rate
		for(int i = 0; i < probVector.size(); i++){
			if(bestVector[i] != worstVector[i]){
				probVector[i] = probVector[i] * (1.0 - negLearningRate) + (bestVector[i] * negLearningRate);
			}
		}


		//mutation during pnil
		int mutateDirection;
		for(int i = 0; i < probVector.size(); i++){

			//gen random
			double random = dis(gen);

			//if random is less than mutprob
			if(random < mutProb){
				random = dis(gen);

				//following pseudocode
				if(random > 0.5){
					mutateDirection = 1;
				}
				else{
					mutateDirection = 0;
				}
				//mutation
				probVector[i] = probVector[i] * (1.0 - .05) + (mutateDirection * .05);
			}
		}


		//just in case
		sampleVector.clear();
		//increase generation
		generation++;
	}
}


int main(int argc, char *argv[]){

	// srand (time(NULL));
	// for(int i = 0; i < 50; i++ ){
	// 	cout << "crossoverPointIndex: " << 2 * ( rand() % ((20-2)/2) ) + 1 << endl;

	// }

	vector<vector<int> > clauseFile;
	string alg = string(argv[7]);
	string filename = argv[1];
	int numIndividuals = atoi(argv[2]);

	if(alg.compare("g") == 0){
		string selectionType = argv[3];
		string crossoverType = argv[4];
		double crossProb = atoi(argv[5]);
		double mutProb = atoi(argv[6]);
		int numGen = atoi(argv[7]);

	}

	if(alg.compare("p") == 0){
		double posLearningRate = atoi(argv[3]);
		double negLearningRate = atoi(argv[4]);
		double mutProb = atoi(argv[5]);
		int numGen = atoi(argv[6]);

	}

	clauseFile = readFile(filename);
	

}



