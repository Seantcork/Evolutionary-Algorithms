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
#include <random>
using namespace std;


int numberOfVariables;

class Individual{

	//are we creating a vector to hold the clauses
	public:
		bool boolVal;
		int fitness = 0;
		void calcFitness(vector<vector<int> > clauseFile);
		// feel free to rename later - KP
		vector<int> negationArray;


};

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

// Changed this from int to void -- let me know if you think this still needs to be an int.
void Individual::calcFitness(vector<vector<int> > clauseFile){

	for(int i = 0; i < clauseFile.size(); i++) {
		for(int j = 0; j < clauseFile.at(i).size(); j++){
			if(clauseFile.at(i).at(j) * this->negationArray.at(clauseFile.at(i).at(j)-1) > 0) {
				this->fitness++;
				break;
			}
		}
	}

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


void tournamentSelection(vector<Individual> Population){

	//manipulate population here or as a global.
	


}


void boltzmanSelection(vector<Individual> Population){



}

vector<Individual> mutatePopulation(double mutProb, vector<Individual> population) {
	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;
   	for(int j = 0; j < population.size(); j++){
   		for(int i = 0; i < population.at(i).negationArray.size(); i++){
		// randDouble(randomEngine) returns a random double between 0 and 1
			if( randDouble(randomEngine) < mutProb ) 
				population.at(i).negationArray.at(i) = population.at(i).negationArray.at(i) * -1;
		}
   	}

   	return population;	

}

// Takes in two parent inviduals and the crossover probability
// Returns a pair of children.
// Haven't tested this function as of yet. 
vector<Individual> onePointCrossover(double crossProb, vector<Individual> population){

	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;
   	Individual firstChild, secondChild, firstParent, secondParent;
   	vector<Individual> returnPopulation;


   	for(int i = 0; i < population.size(); i += 2) {
   		int crossoverPointIndex = 2 * ( rand() % ((firstParent.negationArray.size()-2)/2) ) + 1;

   		// Crossover for first child
   		for(int i = 0; i < firstParent.negationArray.size(); i++) {
   			if(i >= crossoverPointIndex)
   				firstChild.negationArray.push_back(secondParent.negationArray.at(i));
   			else
   				firstChild.negationArray.push_back(firstParent.negationArray.at(i));
   		}

   		// Crossover for second child
   		for(int j = 0; j < secondParent.negationArray.size(); j++) {
   			if(j <= crossoverPointIndex)
   				secondChild.negationArray.push_back(secondParent.negationArray.at(j));
   			else
   				secondChild.negationArray.push_back(firstParent.negationArray.at(j));
   		}
   		// Add children to return population and clear them for the next set of crossover
   		returnPopulation.push_back(firstChild);
   		returnPopulation.push_back(secondChild);
   		firstChild.negationArray.clear();
   		secondChild.negationArray.clear();
   	}

   	return returnPopulation;

   	
}

vector<Individual> uniformCrossover(double crossProb, vector<Individual> population) {

	Individual firstChild, secondChild, firstParent, secondParent;
	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;
   	vector<Individual> returnPopulation

   	for(int i = 0; i < population.size(); i++ ){
   		firstParent = population.at(i);
   		secondParent = population.at(i+1);

   		for(int i = 0; i < firstParent.negationArray.size(); i++) {
			if(randDouble(randomEngine) <= .5) {
				firstChild.negationArray.push_back(firstParent.negationArray.at(i));
			}
			else {
				firstChild.negationArray.push_back(secondParent.negationArray.at(i));
			}
		}

		for(int i = 0; i < firstParent.negationArray.size(); i++) {
			if(randDouble(randomEngine) <= .5) {
				secondChild.negationArray.push_back(firstParent.negationArray.at(i));
			}
			else {
				secondChild.negationArray.push_back(secondParent.negationArray.at(i));
			}

		}
		returnPopulation.push_back(firstChild);
		returnPopulation.push_back(secondChild);
		firstChild.negationArray.clear();
		secondChild.negationArray.clear();
   	}

	return returnPopulation;
}


int genetic_alg(string selectionType, string crossoverType, int numberOfClauses, int numIndividuals, double crossProb, double mutProb, int numGen){
	vector<Individual> population;
	// Initialize population here
	int genCount = 0;

	//have seperate functions for each selection type and run a vertain amount of times
	while(genCount != numGen || population.at(0).fitness != numberOfClauses) {
		population = runSelection(selectionType, population, numIndividuals);

		population = runCrossover(crossoverType, crossProb, population);

		population = mutatePopulation(mutProb, population);

		genCount++;
	}

}

vector<Individual> runSelection(string selectionType, vector<Individual> population, int numIndividuals) {
	if(selectionType == "rs")
		return rankSelection(population, numIndividuals);
	else if(selectionType == "ts")
		return tournamentSelection(population, numIndividuals);
	else if (selectionType == "bs")
		return boltzmanSelection(population, numIndividuals);
}

vector<Individual> runCrossover(string crossoverType, int crossProb, vector<Individual> population ){
	uniform_real_distribution<double> randDouble(0.0,1.0);
   	default_random_engine randomEngine;
   	if(crossProb <= randDouble(randomEngine))
   		if(crossoverType == "1c") {
			return onePointCrossover(crossProb, population);
		else if(crossoverType == "uc")
			return uniformCrossover(crossProb, population);
	else
		return population;
	
	}

}

//returns index of worst solution
vector<int> findBestSolution(vector<Individual> sampleVector, vector<int> evaluations){

	int bestFitness = 0;
	int bestVectorIndex = 0;

	for(int i = 0; i < sampleVector.size(); i++){
		if(evaluations[i] > bestFitness){
			bestFitness = evaluations[i];
			bestVectorIndex = i;
		}
	}

	return sampleVector[bestVectorIndex].negationArray;

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


	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
	

	vector<int> evaluations;
	vector<double> probVector;

	vector<Individual> sampleVector;

	//set normal probabilities
	for(int i =0; i < numberOfVariables; i++){
		probVector[i] = 0.5;
	}

	int generation = 0;
	
	while(generation < numGen){
		for(int i = 0; i < numIndividuals; i ++){
			sampleVector[i] = *generateSampleVector(probVector, numberOfClauses);

			//pretty unsure about this at the moment
			evaluations[i] = evaluate(sampleVector[i], clauseFile);
		}

		vector<int> bestVector;
		vector<int> worstVector;
		bestVector = findBestSolution(sampleVector, evaluations);
		worstVector = findWorstSolution(sampleVector, evaluations);

		for(int i = 0; i < probVector.size(); i++){
			probVector[i] = probVector[i] * (1.0 - posLearningRate) + (bestVector[i] * posLearningRate);

		}

		for(int i = 0; i < probVector.size(); i++){
			if(bestVector[i] != worstVector[i]){
				probVector[i] = probVector[i] * (1.0 - negLearningRate) + (bestVector[i] * negLearningRate);
			}
		}


		int mutateDirection;
		for(int i = 0; i < probVector.size(); i++){
			double random = dis(gen);
			if(random < mutProb){
				random = dis(gen);
				if(random > 0.5){
					mutateDirection = 1;
				}
				else{
					mutateDirection = 0;
				}
				probVector[i] = probVector[i] * (1.0 - .05) + (mutateDirection * .05);
			}
		}
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



