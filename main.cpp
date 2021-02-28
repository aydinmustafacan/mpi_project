/**
Student Name: Mustafa Can AYDIN
Student Number: 2018400303
Compile Status: Compiling
Program Status: Working
Notes: I have written and tested this code in C++11 using MacOS Big Sur version 11.1 using the following commands:
 >mpic++ -o executable ./cmpe300_mpi_2018400303.cpp
 >mpirun --oversubscribe -np number_of_processors executable ./test.tsv
 In case of the error : A system call failed during shared memory initialization that should not have. It is likely that your MPI job will now either abort or experience performance degradation.
 Use below command before mpirun:
 > export TMPDIR=/tmp
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <mpi.h>
#include <stdlib.h>

using namespace std;
#define ABS(a) ((a) < 0 ? - (a) : (a))


double manhatttan(vector<double> first, vector<double> sec){
    double distance_of_manhattan=0;
    for(int i=0; i<first.size();i++){
        double temporary=first[i]-sec[i];
        distance_of_manhattan+= ABS(temporary);
    }
    return distance_of_manhattan;
}

/**
 Helper function in order to help determine Weight array
 */
double diff(int curr_column,int curr_row,int hit_row, vector< vector<double> > &values, double max, double min){
    double res= values[curr_row][curr_column]-values[hit_row][curr_column];
    double denom = max-min;
    res = ABS(res);
    res/=denom;
    return res;

}
//relief algorithm which will return vector of best features
vector<int> relief(int num_of_features, int num_of_iterations, int num_of_sub_instances, double * arr, int num_of_top_features){
    double W[num_of_features];
    vector< vector<double> > values;
    vector<double> tmp;
    for(int i=0 ; i< (num_of_features+1)*num_of_sub_instances; i++){
        tmp.push_back(arr[i]);
        if(tmp.size()%(num_of_features+1)==0){
            values.push_back(tmp);
            tmp.clear();
        }
    }
    //initialize all the weights to zeros
    for(int i=0; i< num_of_features; i++){
        W[i]=0.0;
    }
    //for m iterations do the weight calculation steps
    for(int i=0; i< num_of_iterations; i++){
        int Ri = (i)% num_of_sub_instances;
        //number of sub instance means that each slave has that num of instance to analyze and Ri is random row from values
        //now we need to find H and M values. values[Ri] means : Line with the index Ri
        vector<double> manhattan_distances_bw_same_targets, manhattan_distances_bw_different_targets;
        map<double,int> map_from_distance_to_row_num;
        for(int j=0; j<num_of_sub_instances;j++){
            if(Ri!=j){
                if(values[j][num_of_features]==values[Ri][num_of_features]){
                    double dist= manhatttan(values[Ri], values[j]);
                    manhattan_distances_bw_same_targets.push_back(dist);
                    map_from_distance_to_row_num[dist]=j;
                }
                else if(values[j][num_of_features] != values[Ri][num_of_features]){
                    double dist= manhatttan(values[Ri], values[j]);
                    manhattan_distances_bw_different_targets.push_back(dist);
                    map_from_distance_to_row_num[dist]=j;
                }
            }
        }
        sort(manhattan_distances_bw_same_targets.begin(),manhattan_distances_bw_same_targets.end());
        sort(manhattan_distances_bw_different_targets.begin(),manhattan_distances_bw_different_targets.end());
        int miss_row= map_from_distance_to_row_num[manhattan_distances_bw_different_targets[0]];
        int hit_row = map_from_distance_to_row_num[manhattan_distances_bw_same_targets[0]];
        manhattan_distances_bw_same_targets.clear(); manhattan_distances_bw_different_targets.clear();
        //every feature has different max and min values so we keep the data in vector for each feature
        vector<double> max_values_for_features, min_values_for_features;
        for(int k=0; k<num_of_features;k++){
            double max=values[0][k], min=values[0][k];
            for(int j=0; j<num_of_sub_instances;j++){
                if(max<values[j][k]){
                    max=values[j][k];
                }
                if(min>values[j][k]){
                    min=values[j][k];
                }
            }
            max_values_for_features.push_back(max); min_values_for_features.push_back(min);
        }
        for(int j=0; j<num_of_features;j++){
            double diffH = diff(j, Ri, hit_row,values, max_values_for_features[j], min_values_for_features[j]) ;
            double diffM = diff(j, Ri, miss_row,values, max_values_for_features[j], min_values_for_features[j]);
            W[j]= W[j] - diff(j, Ri, hit_row,values, max_values_for_features[j], min_values_for_features[j])/num_of_iterations+diff(j, Ri, miss_row,values, max_values_for_features[j], min_values_for_features[j])/num_of_iterations;
        }


    }
    int sizeW = sizeof(W)/sizeof(W[0]);
    map<double, int> from_weight_to_index;
    for(int i=0; i<sizeW; i++){
        from_weight_to_index[W[i]] = i ;
    }
    //NOW WE NEED TO RETURN INDEXES OF THE BEST FEATURES
    sort(W, W+sizeW);
    vector<int> bestFeatureVector;
    for(int i=0; i<num_of_top_features; i++){
        bestFeatureVector.push_back(from_weight_to_index[W[sizeW-1-i]]);
    }
    return bestFeatureVector;
}

int main(int argc, char* argv[]) {
    int rank; // rank of the current processor
    int size; // total number of processors
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // gets the total number of processors
    int num_of_instances;
    int num_of_features;
    int num_of_iterations;
    int num_of_top_features;
    int num_of_sub_instances;
    //master processor read the lines and allocates them according to number of slaves to slave processors giving them
    //equal number of data to run relief algorithm on
    int num_of_processor;
    FILE *cin = fopen(argv[1], "r");
    fscanf(cin,"%d", &num_of_processor);//P
    fscanf(cin,"%d", &num_of_instances);//N
    fscanf(cin,"%d", &num_of_features);//A
    if(rank==0){
    fscanf(cin,"%d", &num_of_iterations);//M
    fscanf(cin,"%d", &num_of_top_features);//T
    
    }
    //data of num_of_iterations will be sent by master to slaves
    if(rank==0){
        for(int i=1;i<num_of_processor;i++){
            int data = num_of_iterations;
        MPI_Send(
                /* data         = */ &data,
                /* count        = */ 1,
                /* datatype     = */ MPI_INT,
                /* destination  = */ i,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);
    }
    }
    if(rank!=0){
        int data;
            MPI_Recv(
                    /* data         = */ &data,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* source       = */ 0,
                    /* tag          = */ 0,
                    /* communicator = */ MPI_COMM_WORLD,
                    /* status       = */ MPI_STATUS_IGNORE);
        num_of_iterations=data;
    }
    //data of num_of_top_features will be sent by master to slaves
    if(rank==0){
        for(int i=1;i<num_of_processor;i++){
            int data = num_of_top_features;
            MPI_Send(
                    /* data         = */ &data,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* destination  = */ i,
                    /* tag          = */ 0,
                    /* communicator = */ MPI_COMM_WORLD);
    }
    }
    else if(rank!=0){
        int data;
            MPI_Recv(
                    /* data         = */ &data,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* source       = */ 0,
                    /* tag          = */ 0,
                    /* communicator = */ MPI_COMM_WORLD,
                    /* status       = */ MPI_STATUS_IGNORE);
        num_of_top_features=data;
    }
    num_of_processor--;
    num_of_sub_instances = num_of_instances/ num_of_processor;
    int number_of_total_data = num_of_sub_instances * (num_of_features+1);
    double *arr= new double[number_of_total_data*number_of_total_data+number_of_total_data];
    double pref[number_of_total_data];
    // If it's master processor, reads from input file
    if(rank==0){
        double num=0;
        int j=0,i=number_of_total_data;
        for(;j<number_of_total_data;j++)
            arr[j]=0;
        while(fscanf(cin, "%lf", &num)==1){
            arr[i]=num;
            i++;
        }
        fclose(cin);
    }
    
    // sends data from root array arr to pref array on each processor
    MPI_Scatter(arr,number_of_total_data,MPI_DOUBLE,pref,number_of_total_data,MPI_DOUBLE,0,MPI_COMM_WORLD);
    delete[] arr;
    
    vector<int> ptr_to_W;
    set<double> st;
    //now we need to look at all the features from their weight vector and take the top ones based on their weight value
    if(rank!=0){

        ptr_to_W = relief(num_of_features, num_of_iterations, num_of_sub_instances, pref, num_of_top_features);

        cout <<"Slave P"<<rank<<" :";
        sort(ptr_to_W.begin(), ptr_to_W.end());
        for(int i=0; i<ptr_to_W.size(); i++){
            cout << " "<<ptr_to_W[i];
        } cout << endl;
        
        for(int i=0; i<ptr_to_W.size();i++){
            int data = ptr_to_W[i];
            MPI_Send(
                    /* data         = */ &data,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* destination  = */ 0,
                    /* tag          = */ 0,
                    /* communicator = */ MPI_COMM_WORLD);


        
        }
        
    }
    else if(rank==0){
        for(int j=0;j<num_of_top_features;j++){
            for(int i=1;i<=num_of_processor;i++){
                int data;
                    MPI_Recv(
                            /* data         = */ &data,
                            /* count        = */ 1,
                            /* datatype     = */ MPI_INT,
                            /* source       = */ i,
                            /* tag          = */ 0,
                            /* communicator = */ MPI_COMM_WORLD,
                            /* status       = */ MPI_STATUS_IGNORE);

                    
                st.insert(data);
            }
        }
    }



    int masterSignal = 1;
    while(masterSignal){

        if(rank!= 0){
            int i = 0;
            for(; i<number_of_total_data;i++){
                
            }
        }

        if(rank==0){
            masterSignal=0;
            
        }

        MPI_Bcast(&masterSignal, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast


    }





    //------------------------------------





    //slaves done their job now it's time for the master to unite them all
    //master will take all the vectors from the slaves and add them up to a set
    if(rank==0){
        cout << "Master P0 :";
        for (auto it = st.begin(); it !=
                                   st.end(); ++it)
            cout << " " << *it;
    }


    MPI_Barrier(MPI_COMM_WORLD); // synchronizing processes
    MPI_Finalize();


    return 0;
}
