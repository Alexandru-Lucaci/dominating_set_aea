/*********************************************
 * OPL 22.1.2.0 Model
 * Author: Gurau Iulian
 * Creation Date: Mar 20, 2025 at 9:20:43 PM
 *********************************************/
using CP;

dvar int+ Gas;
dvar int+ Chloride;

maximize 
  40 * Gas + 50 * Chloride;

subject to {
  Gas + Chloride <= 50;  // Capacity constraint
  3 * Gas + 4 * Chloride <= 180;  // Nitrogen + Hydrogen constraint
  Chloride <= 40;  // Chlorine constraint
}
