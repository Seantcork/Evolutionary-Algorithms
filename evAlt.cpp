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
#include <assert.h>     /* assert */
#include <sstream>
#include <random>
#include <math.h>
using namespace std;

const string GENETIC_ALGORITHM = "g";
const string PBIL = "p";

const string RANK_SELECTION = "rs";
const string TOURNAMENT_SELECTION = "ts";
const int TOURNAMENT_SELECTION_M = 2;
const int TOURNAMENT_SELECTION_K = 1;
const string BOLTZMANN_SELECTION = "bs";

const string ONE_POINT_CROSSOVER = "1c";
const string UNIFORM_CROSSOVER = "uc";

int numberOfVariables;
int numberOfClauses;

/*
	Struct that stores all information about the most fit individual the 
	algorithm has found thus far, including its assignments, its fitness, 
	and the generation it was found in.
*/
typedef struct 
	{
		vector<int> bestVector;
		int bestFitness = 0;
		int iteration;
		
	}fittestFoundIndividual;

class Individual{
	/*
		Class to keep track of the individual in a population.

		Attributes:
			varAssignmentArray (vector<int>): A vector that stores whether each
				variable is true (1) or false (-1) for that individual. This is
				implemented as a negation array (using 1s and -1s instead of 0s).
			fitness (int): The fitness of the individual after crossover/mutation.
	*/

	public:
		vector<int> varAssignmentArray;
		int fitness = 0;
		int calcFitness(vector<vector<int> > clauseFile);

};

/*
	Function that evaluates the fitness of the individual. The fitness is determined
	by the number of clauses that the individual has correct in the MAXSAT file. This
	fitness will be calculated after selection/mutation occurs.
*/
int Individual::calcFitness(vector<vector<int> > clauseFile){
	this->fitness = 0;
	bool clauseTrue = false;
	for(int i = 0; i < clauseFile.size(); i++) {

		//checks all elements in a clause
		for(int j = 0; j < clauseFile.at(i).size(); j++){

			//if any element in the clause is true, then the clause is true
			// cout << "Current Variable = " << clauseFile.at(i).at(j) << " negation Array = " << varAssignmentArray.at(abs(clauseFile.at(i).at(j))-1) << endl;
			if(clauseTrue == false && ((clauseFile.at(i).at(j) * this->varAssignmentArray.at(abs(clauseFile.at(i).at(j))-1)) > 0 )){
				this->fitness++;
				clauseTrue = true;
			}
		}
		clauseTrue = false;
	}
	return this->fitness;

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
	string delimiter = " ";
	int position;
	string number;
	for(int i = 0; i < 3; i++){

		position = line.find(delimiter);

		number = line.substr(0, position);

		line.erase(0, position + delimiter.length());

	}
	numberOfVariables = stoi(number);

	position = line.find(delimiter);

	number = line.substr(0, position);

	line.erase(0, position + delimiter.length());
	numberOfClauses = stoi(number);

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

/*
	Function that sorts two individuals in order of increasing fitness.
*/
bool sortPopulation(const Individual & s1, const Individual & s2){
   return s1.fitness < s2.fitness;
}

void printBestVector(vector<int> bestVector){

	for(int i = 0; i < bestVector.size(); i++){
		cout << bestVector.at(i) << endl;
	}
}

/*
	Function that implements rank selection for the Genetic Algorithm. The population
	is sorted by fitness and each individual is given a rank 1 - numIndividuals.
	Probabilities for each individual are created based on these ranks. Then random
	individuals are selected based on these probabilities until the new population reaches
	size numIndividuals.
*/
vector<Individual> rankSelection(vector<Individual> population, int numIndividuals){

	sort(population.begin(), population.end(), sortPopulation);

	/*
		This vector stores the probability each individual has. For example,
		with 4 individuals the vector stores [1, 3, 6, 10] which means the first 
		individual has probability 0 to 1 (10%), the second has 1 to 3 (20%), and
		so on.
	*/
	vector<int> rankProbabilities;
	vector<Individual> breedingPool;

	//rank sum is the total of all ranks, needed to create probabilites from ranks
	int rankSum = numIndividuals * (numIndividuals + 1) * 0.5;

	//the random individual must be between rank probability 1 and the maximum probability
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> gen(1, rankSum);

	//this stores the rank probabilities in the probability vector
	for(int i = 0; i < numIndividuals; i++){
		if(i == 0){
			rankProbabilities.push_back(i + 1);
		}
		else {
			rankProbabilities.push_back(i + 1 + rankProbabilities.at(i-1));
		}
	}

	//random individuals are chosen based on probability, until new population is complete
	for(int i = 0; i < numIndividuals; i++){
		int rand = gen(engine);
		for(int j = 0; j < numIndividuals; j++){

			if(rand <= rankProbabilities.at(j)){
				breedingPool.push_back(population.at(j));
				break;
			}

		}
	}

	return breedingPool;

}

/*
	Function that implements tournament selection for the Genetic Algorithm. The function
	picks m random individuals (in this case 2) and selects the best k (in this case 1). 
	Because the population is sorted we can simply generate two random numbers and choose
	the larger one, which is the index of the fitter individual in the population. This
	repeats until the new population has size numIndividuals.
*/
vector<Individual> tournamentSelection(vector<Individual> population, int numIndividuals){

	sort(population.begin(), population.end(), sortPopulation);
	vector<Individual> breedingPool;

	//the randomly chosen individuals must be within the population index
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> gen(0, numIndividuals-1);

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

/*
	Function that implements Boltzmann selection for the Genetic Algorithm. The function
	assigns probability to each individual based on the boltzmann function (using the
	fitness of the individual rather than its rank). Based on these probabilities, 
	random individuals are chosen until the new population is of size numIndividuals.

	Funtion = e^(fi) / sum(e^fj)
*/
vector<Individual> boltzmannSelection(vector<Individual> population, int numIndividuals){
	vector<Individual> breedingPool;

	//need to calculate the denominator of the boltzmann function (sum of e^fj)
	double sumBolztmannProbabilites = 0;

	for(int i = 0; i < numIndividuals; i++){
		sumBolztmannProbabilites += exp (population.at(i).fitness);
	}

	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_real_distribution<double> gen(0.0, 1.0);

	//vector that stores the probabilities of each individual
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

	//choosing a random individual based on probabilities and adds it to new population
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

/*
	Function that implements one-point crossover for the Genetic Algorithm. The function takes
	in a breeding pool of individuals and performs crossover (based on the inputted crossover
	probability) on pairs of individuals in the breeding pool. The crossover point is randomly
	generated and the two children are added to the new population.
*/
vector<Individual> onePointCrossover(vector<Individual> breedingPool, double crossProb, int numIndividuals){

	std::random_device seeder;
	std::mt19937 engine(seeder());
	//we want to choose a crossover point (that excludes the ends)
	std::uniform_int_distribution<int> gen(1, numberOfVariables - 1);
	std::uniform_int_distribution<double> genDouble(0.0, 1.0);

   	Individual firstChild, secondChild;
   	vector<Individual> newPopulation;

   	//we increase i by 2, because we look at individuals in the breeding pool as pairs (parents)
   	for(int i = 0; i < numIndividuals; i += 2) {

   		//checks whether crossover happens
   		if(genDouble(engine) <= crossProb){

   			//1-point crossover occurs at random index and creates two children
   			int crossoverPointIndex = gen(engine);
   			for(int j = 0; j < numberOfVariables; j++) {
   				if(j >= crossoverPointIndex) {
   					firstChild.varAssignmentArray.push_back(breedingPool.at(i+1).varAssignmentArray.at(j));
   					secondChild.varAssignmentArray.push_back(breedingPool.at(i).varAssignmentArray.at(j));
   				}
   				else {
   					firstChild.varAssignmentArray.push_back(breedingPool.at(i).varAssignmentArray.at(j));
   					secondChild.varAssignmentArray.push_back(breedingPool.at(i+1).varAssignmentArray.at(j));
   				}
   			}

   			//add children to new population
   			newPopulation.push_back(firstChild);
   			newPopulation.push_back(secondChild);
   			firstChild.varAssignmentArray.clear();
   			secondChild.varAssignmentArray.clear();
   		}

   		//if crossover doesn't happen, we keep the individuals from the old population
   		else {
   			newPopulation.push_back(breedingPool.at(i));
			newPopulation.push_back(breedingPool.at(i+1));
   		}
   		
   	}

   	return newPopulation;

}

/*
	Function that implements uniform crossover for the Genetic Algorithm. The function takes
	in a breeding pool of individuals and performs crossover (based on the inputted crossover
	probability) on pairs of individuals in the breeding pool. Each element of the child comes
	randomly from either parent (50% chance each). Because uniform crossover usually only 
	creates one child, we are forcing crossover to occur twice with each parent pair.
*/
vector<Individual> uniformCrossover(vector<Individual> breedingPool, double crossProb, int numIndividuals) {

	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<double> genDouble(0.0, 1.0);

	Individual firstChild, secondChild;
   	vector<Individual> newPopulation;

   	//again increase i by 2, because individuals are treated as pairs
   	for(int i = 0; i < numIndividuals; i += 2){

   		//checks whether crossover happens
   		if(genDouble(engine) <= crossProb){
   			for(int j = 0; j < numberOfVariables; j++) {
   				
   				//creates the first child with 50% chance of each element coming from either parent
				if(genDouble(engine) <= .5){
					firstChild.varAssignmentArray.push_back(breedingPool.at(i).varAssignmentArray.at(j));
				}
				else {
					firstChild.varAssignmentArray.push_back(breedingPool.at(i+1).varAssignmentArray.at(j));
				}

				/* Creates the second child. Since we want the new population size to match old population
			  	   size and uniform crossover only makes 1 child, we will do two crossovers for each pair of
			       parents (gives us two children from each parent).
				*/
				if(genDouble(engine) <= .5){
					secondChild.varAssignmentArray.push_back(breedingPool.at(i).varAssignmentArray.at(j));
				}
				else {
					secondChild.varAssignmentArray.push_back(breedingPool.at(i+1).varAssignmentArray.at(j));
				}

			}

			//children are added to the new population
			newPopulation.push_back(firstChild);
			newPopulation.push_back(secondChild);
			firstChild.varAssignmentArray.clear();
			secondChild.varAssignmentArray.clear();
   		}

   		//if crossover doesn't happen, we keep the individuals from the old population
   		else {
   			newPopulation.push_back(breedingPool.at(i));
			newPopulation.push_back(breedingPool.at(i+1));
   		}

   	}

	return newPopulation;
}

/*
	Function that implements mutation for the Genetic Algorithm. The function does 
	mutation on each element of each individual with the inputted mutation probability.
	If mutation occurs, the element of the individual is flipped (true to false, false
	to true).
*/
vector<Individual> mutatePopulation(vector<Individual> population, double mutProb, int numIndividuals) {
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_real_distribution<double> gen(0.0, 1.0);

   	for(int i = 0; i < numIndividuals; i++){
   		for(int j = 0; j < numberOfVariables; j++){

   			//checks if mutation occurs on the element, if so flips the element
			if(gen(engine) < mutProb){
				population.at(i).varAssignmentArray.at(j) *= -1;
			}
		}
   	}

   	return population;	

}

/*
	Function that implements the Genetic Algorithm. The function initializes a random population
	of size numIndividuals, where the elements of the individuals are randomly generated. The
	algorithm then executes the specified selection, crossover, and mutation for the specified
	number of generations or until a best fit individual is found.
*/
fittestFoundIndividual genetic_alg(vector<vector<int> > clauseFile,
	string selectionType, string crossoverType, int numberOfClauses,
	 int numIndividuals, double crossProb, double mutProb, int numGen){
	
	vector<Individual> population;

	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<double> genDouble(0.0, 1.0);
	
	//initialize the population with random values
	for(int i = 0; i < numIndividuals; i++){
		Individual child;
		for(int j = 0; j < numberOfVariables; j++){

			//1 represents true, -1 is false
			if(genDouble(engine) <= 0.5){
				child.varAssignmentArray.push_back(1);
			}
			else {
				child.varAssignmentArray.push_back(-1);
			}
		}
		population.push_back(child);
	}


	//keeps track of the best individual found so far
	fittestFoundIndividual bestIndividual;


	//loops for the specified number of generations
	int genCount = 1;
	while(genCount <= numGen && population.at(0).fitness != numberOfClauses) {
		
		//evaluate the fitness of each individual, and store if best found individual
		for(int i = 0; i < numIndividuals; i++) {
			if(population.at(i).calcFitness(clauseFile) > bestIndividual.bestFitness){

				bestIndividual.bestFitness = population.at(i).fitness;
				bestIndividual.bestVector = population.at(i).varAssignmentArray;
				bestIndividual.iteration = genCount;
				// assert(bestIndividual.bestFitness <= numberOfClauses);
				if(bestIndividual.bestFitness == numberOfClauses){
					return bestIndividual;
				}

			}
		}

		//does selection based on user input
		if(selectionType.compare(RANK_SELECTION) == 0){
			population = rankSelection(population, numIndividuals);
		}
		else if(selectionType.compare(TOURNAMENT_SELECTION) == 0){
			population = tournamentSelection(population, numIndividuals);
		}
		else if (selectionType.compare(BOLTZMANN_SELECTION) == 0){
			population = boltzmannSelection(population, numIndividuals);
		}

		//does crossover based on user input
		if(crossoverType.compare(ONE_POINT_CROSSOVER) == 0){
			population = onePointCrossover(population, crossProb, numIndividuals);
		}
		else if(crossoverType.compare(UNIFORM_CROSSOVER) == 0){
			population = uniformCrossover(population, crossProb, numIndividuals);
		}

		//does mutation
		population = mutatePopulation(population, mutProb, numIndividuals);

		genCount++;
	}

	return bestIndividual;

}



//returns index of worst solution
Individual findBestSolution(vector<Individual> &sampleVector, vector<vector<int> > clauseFile){

	int bestFitness = 0;
	int bestVectorIndex = -1;

	for(int i = 0; i < sampleVector.size(); i++){
		sampleVector[i].fitness = sampleVector[i].calcFitness(clauseFile);
		if(sampleVector[i].fitness > bestFitness){
			bestFitness = sampleVector[i].fitness;
			bestVectorIndex = i;
		}
	}
	return sampleVector[bestVectorIndex];

}

//returns index of best solution
Individual findWorstSolution(vector<Individual> sampleVector, vector<vector<int> > clauseFile ){


	int worstFitness = 8000;
	int worstFitnessIndex = 0;

	for(int i = 0; i < sampleVector.size(); i++){
		sampleVector[i].calcFitness(clauseFile);
		if(sampleVector[i].fitness < worstFitness){
			worstFitness = sampleVector[i].fitness;
			worstFitnessIndex = i;
		}
	}

	return sampleVector[worstFitnessIndex];

}


//generates a vector of bools for use in a pbil
Individual generateSampleVector(vector<double> probVector, int numberOfClauses){

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Individual indv;
    
	for(int i = 0; i < probVector.size(); i ++){

		//probability for chosing true or false;
		double probForUse = dis(gen);


		//if the porbability is useful than its one
		if(probForUse <= probVector[i]){
			//cout << "1" << endl;
			
			indv.varAssignmentArray.push_back(1);
		}

		//if the probability is not than its zero
		else{
			//cout << "2" << endl;
			indv.varAssignmentArray.push_back(0);
		}
	}
	
	return indv;
	//initialize the population with random values
}





fittestFoundIndividual pbil(vector<vector<int> > clauseFile, int numberOfClauses,
 int numIndividuals, int posLearningRate, int negLearningRate, double mutProb,int mutationAmount, int numGen){

	//helps us keep track of best individual so far

	//create struct to keep track of bestIndividual
	fittestFoundIndividual bestIndividual;

	//best individual from each roun

	Individual child;
	
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
	
	//probability vector
	vector<double> probVector;


	//best vector we have found so far
	vector<int> bestVector;
	vector<int> worstVector;
	//the fitness of best vector

	//vector used for sample population
	vector<Individual> sampleVector;
	//set normal probabilities
	for(int i = 0; i < numberOfVariables; i++){
		probVector.push_back(.5);
	}
	//generation nuymber
	int generation = 0;
	while(generation < numGen){

		for(int i = 0; i < numIndividuals; i++){
			Individual child;
			for(int j = 0; j < numberOfVariables; j++){

			//1 represents true, -1 is false
				if(dis(gen) <= probVector[j]){
					child.varAssignmentArray.push_back(1);
				}
				else {
					child.varAssignmentArray.push_back(-1);
				}
			}
			sampleVector.push_back(child);
		}

		Individual bestInRound;
		Individual worstInRound;

		//pass data into struct that keeps track of best solutions

		bestInRound = findBestSolution(sampleVector, clauseFile);
		bestVector = bestInRound.varAssignmentArray;

		if(bestInRound.fitness > bestIndividual.bestFitness){
			bestIndividual.bestFitness = bestInRound.fitness;
			bestIndividual.bestVector = bestInRound.varAssignmentArray;
		}

		//gives us the best vector from this iteration
		//bestVector = bestRound.bestVector;
		worstInRound = findWorstSolution(sampleVector, clauseFile);
		worstVector = worstInRound.varAssignmentArray;

		//keep track of best individual we have found

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
				probVector[i] = probVector[i] * (1.0 - mutationAmount) + (mutateDirection * mutationAmount);
			}
		}
		//just in case
		sampleVector.clear();
		//increase generation
		generation++;
	}
	return bestIndividual;
}


int main(int argc, char *argv[]){

	fittestFoundIndividual result;
	vector<vector<int> > clauseFile;
	string alg = string(argv[8]);
	string filename = argv[1];
	int numIndividuals = atoi(argv[2]);
	clauseFile = readFile(filename);

	if(alg.compare(GENETIC_ALGORITHM) == 0){
		string selectionType = argv[3];
		string crossoverType = argv[4];
		double crossProb = atoi(argv[5]);
		double mutProb = atoi(argv[6]);
		int numGen = atoi(argv[7]);

		result = genetic_alg(clauseFile, selectionType, crossoverType, numberOfClauses, numIndividuals, crossProb, mutProb, numGen);

	}

	if(alg.compare(PBIL) == 0){
		double posLearningRate = atoi(argv[3]);
		double negLearningRate = atoi(argv[4]);
		double mutProb = atoi(argv[5]);

		double mutationAmount= atoi(argv[6]);

		int numGen = atoi(argv[7]);


		result = pbil(clauseFile, numberOfClauses, numIndividuals, posLearningRate, negLearningRate, mutProb, mutationAmount, numGen);

	}

	//clauseFile = readFile(filename);
	double fitnessPercent = (double(result.bestFitness) / double(numberOfClauses)) * 100;

	cout << filename << endl;
	cout << numberOfVariables << endl;
	cout << "number of clauses: " << numberOfClauses << endl;
	cout << "Result Best Fitness: " << result.bestFitness << endl;
	cout << "Fitness Percent: " << fitnessPercent << "% " << endl;
	// printBestVector(result.bestVector);
	cout << result.iteration << endl;
	

}



