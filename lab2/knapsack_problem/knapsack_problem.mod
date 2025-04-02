/*********************************************
 * OPL 22.1.2.0 Model
 * Author: Gurau Iulian
 * Creation Date: Mar 20, 2025 at 9:25:20 PM
 *********************************************/
using CP;

//// Number of items
//int n;  
//
//// Knapsack capacity
//int capacity;  
//
//// Arrays for weights and values of items
//int weights[1..n];
//int values[1..n];

int n = 4;  // Number of items
int capacity = 10;  // Knapsack weight limit

int weights[1..n] = [2, 3, 4, 5];  // Item weights
int values[1..n]  = [3, 4, 5, 6];  // Item values

// Decision variables: Number of each item to take
dvar int+ x[1..n];

// Objective: Maximize the total value
maximize sum(i in 1..n) values[i] * x[i];

subject to {
    // Total weight should not exceed capacity
    sum(i in 1..n) weights[i] * x[i] <= capacity;
}